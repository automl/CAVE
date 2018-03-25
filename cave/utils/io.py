from bokeh.io import export_png

def export_bokeh(plot, path, logger):
    logger.debug("Exporting to %s", path)
    try:
        export_png(plot, filename=path)
    except (RuntimeError, TypeError) as err:
        logger.debug("Exporting failed with message \"%s\"", err)
        logger.warning("To activate png-export, please follow "
                       "instructions on CAVE's GitHub (install "
                       "selenium and phantomjs-prebuilt).")
