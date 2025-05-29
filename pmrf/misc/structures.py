import skrf as rf

def frequency_to_dict(frequency: rf.Frequency) -> dict:
    """Convert skrf Frequency object to a dictionary."""
    return {
        'start': frequency.start_scaled,
        'stop': frequency.stop_scaled,
        'npoints': frequency.npoints,
        'unit': frequency.unit,
    }
    
def dict_to_frequency(freq_dict: dict) -> rf.Frequency:
    """Convert a dictionary to skrf Frequency object."""
    return rf.Frequency(
        start=freq_dict['start'],
        stop=freq_dict['stop'],
        npoints=freq_dict['npoints'],
        unit=freq_dict['unit']
    )
    
class ObservableDict(dict):
    """Custom dictionary that triggers a callback on updates."""
    def __init__(self, owner=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.owner = owner
        self.notify = True
        self.update_callbacks = []
        self.set_callbacks = []
        self.get_callbacks = []

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._call_set_callbacks(key, value)

    def __getitem__(self, key):
        self._call_get_callbacks(key)
        return super().__getitem__(key)

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._call_update_callbacks()

    def add_update_callback(self, callback):
        if callback not in self.update_callbacks:
            self.update_callbacks.insert(0, callback)

    def add_set_callback(self, callback):
        if callback not in self.set_callbacks:
            self.set_callbacks.insert(0, callback)
        
    def add_get_callback(self, callback):
        if callback not in self.get_callbacks:
            self.get_callbacks.insert(0, callback)

    def remove_update_callback(self, callback):
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)

    def remove_set_callback(self, callback):    
        if callback in self.set_callbacks:
            self.set_callbacks.remove(callback)

    def remove_get_callback(self, callback):    
        if callback in self.get_callbacks:
            self.get_callbacks.remove(callback)

    def remove_callbacks(self):
        self.update_callbacks = []
        self.set_callbacks = []
        self.get_callbacks = []

    def set_notify(self, notify=False):
        self.notify = notify

    def _call_update_callbacks(self):
        if not self.notify:
            return
        for callback in self.update_callbacks:
            callback(self)

    def _call_set_callbacks(self, key, value):
        if not self.notify:
            return
        for callback in self.set_callbacks:
            callback(key, value, self)

    def _call_get_callbacks(self, key):
        if not self.notify:
            return
        for callback in self.get_callbacks:
            callback(key, self)