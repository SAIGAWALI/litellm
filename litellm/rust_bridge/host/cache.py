from __future__ import annotations

import time
import weakref
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Final, cast  # noqa: TID251  # bounded casts of the legacy handler and response

from pydantic import TypeAdapter

from litellm.litellm_core_utils.add_retry_fallback_headers import get_hidden_params_dict

if TYPE_CHECKING:
    from litellm.caching.caching import Cache
    from litellm.caching.caching_handler import LLMCachingHandler
    from litellm.litellm_core_utils.litellm_logging import Logging

_RUST_RESPONSE_HEADER: Final = "x-litellm-rust"
_HEADERS: Final = TypeAdapter(dict[str, object])
_HITS: Final[weakref.WeakKeyDictionary[Logging, Mapping[str, object]]] = weakref.WeakKeyDictionary()


def configured_cache() -> Cache | None:
    import litellm

    return litellm.cache


def _python_entry(response: Mapping[str, object]) -> Mapping[str, object]:
    hidden: Final = get_hidden_params_dict(response)
    headers: Final = hidden.get("additional_headers")
    if not isinstance(headers, Mapping) or _RUST_RESPONSE_HEADER not in headers:
        return response
    kept: Final = {
        name: value for name, value in _HEADERS.validate_python(headers).items() if name != _RUST_RESPONSE_HEADER
    }
    stored: Final = {name: value for name, value in hidden.items() if name != "additional_headers" or kept} | (
        {"additional_headers": kept} if kept else {}
    )
    entry: Final = {name: value for name, value in response.items() if name != "_hidden_params"}
    return {**entry, "_hidden_params": stored} if stored else entry


def store_response(
    cache: Cache,
    response: Mapping[str, object],
    /,
    **kwargs: object,  # kwargs-ok: the cache call's own arguments
) -> None:
    from litellm.caching.caching_handler import create_cache_write_task

    create_cache_write_task(
        lambda: cache.async_add_cache(  # pyright: ignore[reportUnknownMemberType]  # Cache.async_add_cache leaves its result untyped
            _python_entry(response),
            **kwargs,  # pyright: ignore[reportArgumentType]  # the cache call's own keyword arguments, as Python passes them
        )
    )


async def lookup(
    _cache: Cache,
    /,
    **kwargs: object,  # kwargs-ok: the cache call's own arguments
) -> object | None:
    from litellm.caching.caching_handler import LLMCachingHandler
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.llms.anthropic.messages.native import handler as messages

    original_function: Final = cast(  # cast-ok: the @client cache read names and checks the legacy handler it wraps
        "Callable[..., object]",
        messages.anthropic_messages,  # noqa: TID251  # the @client cache read names and checks the legacy handler it wraps
    )

    logging_obj: Final = kwargs.get("litellm_logging_obj")
    if not isinstance(logging_obj, Logging):
        return None
    handler: Final = LLMCachingHandler(
        original_function=original_function, request_kwargs=dict(kwargs), start_time=logging_obj.start_time
    )
    logging_obj.llm_caching_handler = handler
    cached: Final[object] = await _cached_response(handler, logging_obj, kwargs, original_function)
    _remember(logging_obj, cached)
    return cached


def _remember(logging_obj: Logging, cached: object) -> None:
    if isinstance(cached, Mapping):
        _HITS[logging_obj] = _HEADERS.validate_python(cached)


def _replace(response: object, cached: Mapping[str, object] | None) -> None:
    if isinstance(response, dict) and cached is not None:
        served: Final = cast(  # cast-ok: a Messages response is the str-keyed dict the route serves
            "dict[str, object]", response
        )
        served.clear()
        served.update(cached)


def restore_hit(response: object, logger: Logging) -> None:
    from litellm.caching.caching_handler import (
        _set_cached_hidden_param,  # pyright: ignore[reportPrivateUsage]  # the @client cache hit's own hidden params
    )

    _replace(response, _HITS.pop(logger, None))
    _set_cached_hidden_param(response, "cache_hit", True)


def hit_cache_key(logger: Logging) -> object:
    import litellm
    from litellm.caching.caching_handler import LLMCachingHandler

    handler: Final = logger.llm_caching_handler
    cache: Final = litellm.cache
    if not isinstance(handler, LLMCachingHandler) or cache is None:
        return None
    return (
        handler.preset_cache_key
        or handler.request_kwargs.get("cache_key")
        or cache.get_cache_key(**handler.request_kwargs)  # pyright: ignore[reportUnknownMemberType]  # Cache.get_cache_key takes the call's own keyword arguments
    )


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


async def _cached_response(
    handler: LLMCachingHandler,
    logging_obj: Logging,
    arguments: Mapping[str, object],
    original_function: Callable[..., object],
) -> object | None:
    import litellm
    from litellm.caching.caching_handler import (
        _is_response_cache_excluded,  # pyright: ignore[reportPrivateUsage]  # the @client cache read's own exclusion
    )
    from litellm.litellm_core_utils.core_helpers import get_parent_otel_span_from_kwargs

    model: Final = arguments.get("model")
    model_name: Final = model if isinstance(model, str) else ""
    if _is_response_cache_excluded(model=model_name, kwargs=arguments):
        return None
    kwargs: Final = dict(arguments)
    kwargs["parent_otel_span"] = get_parent_otel_span_from_kwargs(kwargs)
    if not handler._is_call_type_supported_by_cache(  # pyright: ignore[reportPrivateUsage]  # the @client cache read's own check
        original_function=original_function
    ):
        return None
    logging_obj.reset_caching_details()
    started: Final = time.perf_counter()
    cached: Final = await handler._retrieve_from_cache(  # pyright: ignore[reportPrivateUsage]  # the @client cache read
        call_type="anthropic_messages", kwargs=kwargs, args=()
    )
    finished: Final = time.perf_counter()
    if cached is None or isinstance(cached, list):
        return None
    provider_model, custom_llm_provider, _, _ = litellm.get_llm_provider(
        model=model_name,
        custom_llm_provider=_optional_str(kwargs.get("custom_llm_provider")),
        api_base=_optional_str(kwargs.get("api_base")),
        api_key=_optional_str(kwargs.get("api_key")),
    )
    handler._update_litellm_logging_obj_environment(  # pyright: ignore[reportPrivateUsage]  # the @client cache hit's logging update
        logging_obj=logging_obj,
        model=provider_model,
        kwargs=kwargs,
        cached_result=cached,
        is_async=True,
        custom_llm_provider=custom_llm_provider,
        cache_duration_ms=(finished - started) * 1000,
    )
    return cached
