# 
# import _pupillometry

def _check_pyeparse():
    """Helper to ensure package is available"""
    try:
        import pyeparse  # noqa analysis:ignore
    except ImportError:
        raise ImportError('Cannot run, requires "pyeparse" package')

def _load_raw(el, fname):
    """Helper to load some pupil data"""
    import pyeparse
    fname = el.transfer_remote_file(fname)
    # Load and parse data
    logger.info('Pupillometry: Parsing local file "{0}"'.format(fname))
    raw = pyeparse.Raw(fname)
    raw.remove_blink_artifacts()
    events = raw.find_events('SYNCTIME', 1)
    return raw, events
