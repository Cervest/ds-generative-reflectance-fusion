from src.utils import Registry
"""
Registery of common callbacks
"""
CALLBACKS = Registry()


def build_callback(cfg):
    experiment = CALLBACKS[cfg['name']](cfg)
    return experiment


@CALLBACKS.register('early_stopping')
def build_early_stopping(cfg):
    """Builds early stopping callback
    """
    if cfg:
        import pytorch_lightning as pl
        params = {'monitor': cfg['monitor'],
                  'patience': cfg['patience']}
        callback = pl.callbacks.EarlyStopping(**params)
    else:
        callback = False
    return callback
