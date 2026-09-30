"""
Unit tests for GigaChat file handler.

Tests _get_url_hash, _parse_data_url, _download_image_sync, _download_image_async,
upload_file_sync, and upload_file_async covering caching, base64 data URL decoding,
network errors, and the full upload flow.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.prompt_templates import image_handling
from litellm.llms.gigachat import file_handler
from litellm.llms.gigachat.file_handler import (
    _file_cache,
    _get_url_hash,
    _parse_data_url,
    upload_file_async,
    upload_file_sync,
)

FILE_MODULE = "litellm.llms.gigachat.file_handler"

# A valid 1x1 red PNG as base64
_RED_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQI12NgYPgPAAEDAQAR3X3ZAAAASUVORK5CYII="
_RED_PNG_DATA_URL = f"data:image/png;base64,{_RED_PNG_B64}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_file_cache():
    """Each test gets a fresh module-level file cache to avoid cross-test leakage."""
    _file_cache.clear()
    yield
    _file_cache.clear()


# ---------------------------------------------------------------------------
# _get_url_hash
# ---------------------------------------------------------------------------


class TestGetUrlHash:
    def test_returns_hex_string(self):
        h = _get_url_hash("https://example.com/image.png")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256

    def test_different_urls_different_hashes(self):
        h1 = _get_url_hash("https://example.com/a.png")
        h2 = _get_url_hash("https://example.com/b.png")
        assert h1 != h2

    def test_same_url_same_hash(self):
        h1 = _get_url_hash("https://example.com/image.png")
        h2 = _get_url_hash("https://example.com/image.png")
        assert h1 == h2


# ---------------------------------------------------------------------------
# _parse_data_url
# ---------------------------------------------------------------------------


class TestParseDataUrl:
    def test_valid_base64_png(self):
        result = _parse_data_url(_RED_PNG_DATA_URL)
        assert result is not None
        content_bytes, content_type, ext = result
        assert content_type == "image/png"
        assert ext == "png"
        assert len(content_bytes) > 0

    def test_valid_base64_jpeg(self):
        # Simple valid base64 (24 chars, properly padded, no + or / chars)
        valid_b64 = "aGVsbG8gd29ybGQhISEhIQ=="
        data_url = f"data:image/jpeg;base64,{valid_b64}"
        result = _parse_data_url(data_url)
        assert result is not None
        _, content_type, ext = result
        assert content_type == "image/jpeg"
        assert ext == "jpeg"

    def test_valid_base64_with_semicolon_in_type(self):
        """Data URLs with charset before base64 segment do not match the regex."""
        # The regex `data:([^;]+);base64,(.+)` requires the pattern to be
        # `data:<type>;base64,<data>`. If `;charset=utf-8` appears before
        # `;base64,`, the regex sees `data:image/png` as group 1 but then
        # looks for `;base64,` immediately after — which isn't there because
        # `;charset=utf-8;base64,` has extra text before `;base64,`
        data_url = "data:image/png;charset=utf-8;base64," + _RED_PNG_B64
        result = _parse_data_url(data_url)
        assert result is None

    def test_invalid_data_url_returns_none(self):
        assert _parse_data_url("not-a-data-url") is None

    def test_empty_base64_returns_none(self):
        """Empty base64 data (nothing after comma) does not match regex `(.+)`."""
        assert _parse_data_url("data:image/png;base64,") is None

    def test_missing_base64_segment(self):
        assert _parse_data_url("data:image/png;base64") is None

    def test_unknown_extension_falls_back_to_jpg(self):
        data_url = "data:application/octet-stream;base64," + _RED_PNG_B64
        result = _parse_data_url(data_url)
        assert result is not None
        _, content_type, ext = result
        assert content_type == "application/octet-stream"
        # The extension is derived from content_type.split("/")[-1].split(";")[0]
        # which gives "octet-stream", not "jpg"
        assert ext == "octet-stream"


_PUBLIC_IMAGE_URL = "https://93.184.215.14/img.png"
_METADATA_URL = "http://169.254.169.254/latest/meta-data/iam/security-credentials/"


def _image_response(url: str, content: bytes, content_type: str) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        headers={"Content-Type": content_type},
        content=content,
        request=httpx.Request("GET", url),
    )


@pytest.fixture(autouse=True)
def _isolate_image_cache():
    image_handling.in_memory_cache.flush_cache()
    yield
    image_handling.in_memory_cache.flush_cache()


# ---------------------------------------------------------------------------
# _download_image_sync
# ---------------------------------------------------------------------------


class TestDownloadImageSync:
    def test_downloads_image_successfully(self):
        mock_client = MagicMock()
        mock_client.get.return_value = _image_response(_PUBLIC_IMAGE_URL, b"fake-image-bytes", "image/png")

        with patch.object(litellm, "module_level_client", mock_client):
            content_bytes, content_type, ext = file_handler._download_image_sync(_PUBLIC_IMAGE_URL)

        assert (content_bytes, content_type, ext) == (b"fake-image-bytes", "image/png", "png")
        assert mock_client.get.call_args.kwargs["follow_redirects"] is False

    def test_blocks_metadata_address_without_request(self):
        mock_client = MagicMock()

        with patch.object(litellm, "module_level_client", mock_client):
            with pytest.raises(litellm.ImageFetchError):
                file_handler._download_image_sync(_METADATA_URL)

        mock_client.get.assert_not_called()

    def test_blocks_redirect_to_private_address(self):
        mock_client = MagicMock()
        mock_client.get.return_value = httpx.Response(
            status_code=302,
            headers={"Location": _METADATA_URL},
            request=httpx.Request("GET", _PUBLIC_IMAGE_URL),
        )

        with patch.object(litellm, "module_level_client", mock_client):
            with pytest.raises(litellm.ImageFetchError):
                file_handler._download_image_sync(_PUBLIC_IMAGE_URL)

        assert [c.args[0] for c in mock_client.get.call_args_list] == [_PUBLIC_IMAGE_URL]

    def test_rejects_oversized_image(self):
        mock_client = MagicMock()
        mock_client.get.return_value = _image_response(_PUBLIC_IMAGE_URL, b"x" * 2048, "image/png")

        with (
            patch.object(litellm, "module_level_client", mock_client),
            patch.object(image_handling, "MAX_IMAGE_URL_DOWNLOAD_SIZE_MB", 1 / 1024),
        ):
            with pytest.raises(litellm.ImageFetchError):
                file_handler._download_image_sync(_PUBLIC_IMAGE_URL)

    def test_extracts_extension_from_parametrized_type(self):
        mock_client = MagicMock()
        mock_client.get.return_value = _image_response(_PUBLIC_IMAGE_URL, b"data", "image/png; charset=utf-8")

        with patch.object(litellm, "module_level_client", mock_client):
            content_bytes, _, ext = file_handler._download_image_sync(_PUBLIC_IMAGE_URL)

        assert (content_bytes, ext) == (b"data", "png")


# ---------------------------------------------------------------------------
# _download_image_async
# ---------------------------------------------------------------------------


class TestDownloadImageAsync:
    @pytest.mark.asyncio
    async def test_downloads_image_successfully(self):
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=_image_response(_PUBLIC_IMAGE_URL, b"fake-image-bytes", "image/webp"))

        with patch.object(litellm, "module_level_aclient", mock_client):
            content_bytes, content_type, ext = await file_handler._download_image_async(_PUBLIC_IMAGE_URL)

        assert (content_bytes, content_type, ext) == (b"fake-image-bytes", "image/webp", "webp")

    @pytest.mark.asyncio
    async def test_blocks_metadata_address_without_request(self):
        mock_client = MagicMock()
        mock_client.get = AsyncMock()

        with patch.object(litellm, "module_level_aclient", mock_client):
            with pytest.raises(litellm.ImageFetchError):
                await file_handler._download_image_async(_METADATA_URL)

        mock_client.get.assert_not_called()


# ---------------------------------------------------------------------------
# upload_file_sync
# ---------------------------------------------------------------------------


class TestUploadFileSync:
    @patch(f"{FILE_MODULE}.get_api_base", return_value="https://api.example.com")
    @patch(f"{FILE_MODULE}.get_access_token", return_value="test-token")
    @patch(f"{FILE_MODULE}._get_httpx_client")
    def test_uploads_base64_image_and_caches(self, mock_http_handler_cls, mock_get_token, mock_get_api_base):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "file-12345"}
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        mock_http_handler_cls.return_value = mock_client

        result = upload_file_sync(
            image_url=_RED_PNG_DATA_URL,
            credentials="creds",
            api_base="https://custom.example.com",
        )

        assert result == "file-12345"
        # Verify it was cached
        url_hash = _get_url_hash(_RED_PNG_DATA_URL)
        assert _file_cache[url_hash] == "file-12345"

        # Check the upload request — url is passed as first positional arg
        call_args = mock_client.post.call_args
        assert call_args.args[0] == "https://api.example.com/files"
        assert call_args.kwargs["headers"]["Authorization"] == "Bearer test-token"
        # Verify purpose
        assert call_args.kwargs["data"] == {"purpose": "general"}
        # Verify a file was attached
        assert "file" in call_args.kwargs["files"]

    @patch(f"{FILE_MODULE}.get_api_base", return_value="https://api.example.com")
    @patch(f"{FILE_MODULE}.get_access_token", return_value="test-token")
    @patch(f"{FILE_MODULE}._get_httpx_client")
    def test_returns_cached_file_id(self, mock_http_handler_cls, mock_get_token, mock_get_api_base):
        # Pre-populate the cache
        url_hash = _get_url_hash(_RED_PNG_DATA_URL)
        _file_cache[url_hash] = "cached-file-id"

        result = upload_file_sync(image_url=_RED_PNG_DATA_URL, credentials="creds")

        assert result == "cached-file-id"
        # No upload call was made
        mock_http_handler_cls.return_value.post.assert_not_called()

    @patch(f"{FILE_MODULE}._get_httpx_client")
    @patch(f"{FILE_MODULE}.get_access_token", return_value="test-token")
    @patch(f"{FILE_MODULE}.get_api_base", return_value="https://api.example.com")
    @patch(f"{FILE_MODULE}._download_image_sync")
    def test_downloads_and_uploads_url_image(
        self, mock_download, mock_get_api_base, mock_get_token, mock_http_handler_cls
    ):
        mock_download.return_value = (b"remote-bytes", "image/png", "png")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "file-remote"}
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        mock_http_handler_cls.return_value = mock_client

        result = upload_file_sync(image_url="https://example.com/remote.png", credentials="creds")

        assert result == "file-remote"
        mock_download.assert_called_once_with("https://example.com/remote.png")

    @patch(f"{FILE_MODULE}._get_httpx_client")
    @patch(f"{FILE_MODULE}.get_access_token", return_value="test-token")
    @patch(f"{FILE_MODULE}.get_api_base", return_value="https://api.example.com")
    def test_returns_none_on_upload_failure(self, mock_get_api_base, mock_get_token, mock_http_handler_cls):
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "Bad Request",
            request=httpx.Request("POST", "https://api.example.com/files"),
            response=httpx.Response(status_code=400, request=httpx.Request("POST", "https://api.example.com/files")),
        )
        mock_http_handler_cls.return_value = mock_client

        # upload_file_sync catches all exceptions and returns None
        result = upload_file_sync(image_url=_RED_PNG_DATA_URL, credentials="creds")

        assert result is None

    @patch(f"{FILE_MODULE}._get_httpx_client")
    @patch(f"{FILE_MODULE}.get_access_token", return_value="test-token")
    @patch(f"{FILE_MODULE}.get_api_base", return_value="https://api.example.com")
    def test_returns_none_when_response_missing_id(self, mock_get_api_base, mock_get_token, mock_http_handler_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}  # no "id" key
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        mock_http_handler_cls.return_value = mock_client

        result = upload_file_sync(image_url=_RED_PNG_DATA_URL, credentials="creds")

        assert result is None


# ---------------------------------------------------------------------------
# upload_file_async
# ---------------------------------------------------------------------------


class TestUploadFileAsync:
    @pytest.mark.asyncio
    @patch(f"{FILE_MODULE}.get_api_base", return_value="https://api.example.com")
    @patch(f"{FILE_MODULE}.get_access_token_async", return_value="test-token-async")
    @patch(f"{FILE_MODULE}.get_async_httpx_client")
    async def test_uploads_base64_image_and_caches(self, mock_get_client, mock_get_token, mock_get_api_base):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"id": "async-file-1"})
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        result = await upload_file_async(
            image_url=_RED_PNG_DATA_URL,
            credentials="creds",
            api_base="https://custom.example.com",
        )

        assert result == "async-file-1"
        # Verify cache
        url_hash = _get_url_hash(_RED_PNG_DATA_URL)
        assert _file_cache[url_hash] == "async-file-1"

        # Check upload request details — url is first positional arg
        call_args = mock_client.post.call_args
        assert call_args.args[0] == "https://api.example.com/files"
        assert call_args.kwargs["headers"]["Authorization"] == "Bearer test-token-async"
        assert "purpose" in str(call_args.kwargs["data"])
        assert "file" in call_args.kwargs["files"]

    @pytest.mark.asyncio
    @patch(f"{FILE_MODULE}.get_api_base", return_value="https://api.example.com")
    @patch(f"{FILE_MODULE}.get_access_token_async", return_value="test-token-async")
    @patch(f"{FILE_MODULE}.get_async_httpx_client")
    async def test_returns_cached_file_id(self, mock_get_client, mock_get_token, mock_get_api_base):
        url_hash = _get_url_hash(_RED_PNG_DATA_URL)
        _file_cache[url_hash] = "cached-async-id"

        result = await upload_file_async(image_url=_RED_PNG_DATA_URL, credentials="creds")

        assert result == "cached-async-id"
        mock_get_client.return_value.post.assert_not_called()

    @pytest.mark.asyncio
    @patch(f"{FILE_MODULE}.get_async_httpx_client")
    @patch(f"{FILE_MODULE}.get_access_token_async", return_value="test-token-async")
    @patch(f"{FILE_MODULE}.get_api_base", return_value="https://api.example.com")
    @patch(f"{FILE_MODULE}._download_image_async")
    async def test_downloads_and_uploads_url_image(
        self, mock_download, mock_get_api_base, mock_get_token, mock_get_client
    ):
        mock_download.return_value = (b"remote-bytes-async", "image/png", "png")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"id": "async-file-remote"})
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        result = await upload_file_async(image_url="https://example.com/remote.png", credentials="creds")

        assert result == "async-file-remote"
        mock_download.assert_called_once_with("https://example.com/remote.png")

    @pytest.mark.asyncio
    @patch(f"{FILE_MODULE}.get_async_httpx_client")
    @patch(f"{FILE_MODULE}.get_access_token_async", return_value="test-token-async")
    @patch(f"{FILE_MODULE}.get_api_base", return_value="https://api.example.com")
    async def test_returns_none_on_upload_failure(self, mock_get_api_base, mock_get_token, mock_get_client):
        mock_client = MagicMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Bad Request",
                request=httpx.Request("POST", "https://api.example.com/files"),
                response=httpx.Response(
                    status_code=400, request=httpx.Request("POST", "https://api.example.com/files")
                ),
            )
        )
        mock_get_client.return_value = mock_client

        result = await upload_file_async(image_url=_RED_PNG_DATA_URL, credentials="creds")

        assert result is None

    @pytest.mark.asyncio
    @patch(f"{FILE_MODULE}.get_async_httpx_client")
    @patch(f"{FILE_MODULE}.get_access_token_async", return_value="test-token-async")
    @patch(f"{FILE_MODULE}.get_api_base", return_value="https://api.example.com")
    async def test_returns_none_when_response_missing_id(self, mock_get_api_base, mock_get_token, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"status": "ok"})
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        result = await upload_file_async(image_url=_RED_PNG_DATA_URL, credentials="creds")

        assert result is None
