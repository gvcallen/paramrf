def format_val(val, sig_figs: int = 4) -> str:
    """Safely extract and format a scalar or array value for clean printing."""
    if hasattr(val, "ndim") and val.ndim == 0:
        return f"{float(val):.{sig_figs}g}"
    elif hasattr(val, "size") and val.size == 1:
        return f"{float(val.item()):.{sig_figs}g}"
    else:
        # Fallback for multi-dimensional arrays
        return f"f64{list(val.shape)}" if hasattr(val, "shape") else repr(val)