def apt_warning(logger):
    """
    This function only exists to clarify, that APT-support is still work in progress
    """
    logger.warning("Attention! Auto-PyTorch support is still WIP, since APT itself is still an early alpha. "
                   "Please report errors and issues to https://github.com/automl/CAVE/issues with a MWE."
                   "This statement holds for CAVE 1.3.4. The last working branch of APT was "
                   "https://github.com/automl/Auto-PyTorch/commit/a39012ff464a02eead9315a00179812206235f25.")