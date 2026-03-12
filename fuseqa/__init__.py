_SRE_IMPORT_ERROR = None

try:
    from .sre import SRE
except ModuleNotFoundError as exc:
    _SRE_IMPORT_ERROR = exc

    class SRE:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "SRE dependencies are missing. Install them to use fuseqa.SRE "
                "(e.g. pip install prettytable)."
            ) from _SRE_IMPORT_ERROR

from .utils import (
    hf_model_to_filename,
    ask_llm_generate,
    parse_list,
    build_context,
    tier_from_spop,
    clean_pred,
    safe_div,
    norm,
    generate_report,
    write_record,
    build_context_from_sre_list,
    get_context_for_run
    
)

__all__ = (
    "SRE",
    "hf_model_to_filename",
    "ask_llm_generate",
    "parse_list",
    "build_context",
    "tier_from_spop",
    "clean_pred",
    "safe_div",
    "norm",
    "generate_report",
    "write_record",
    "build_context_from_sre_list",
    "get_context_for_run",
)
