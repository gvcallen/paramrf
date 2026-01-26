import io
import base64
from typing import BinaryIO, TypeVar
import tempfile

from numpyro.distributions import Distribution
    
class SerializableDistribution(Distribution):
    def write(self, target: BinaryIO):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=True) as tmp:
            self.save(tmp.name)
            tmp.seek(0)
            target.write(tmp.read())
    
    @classmethod
    def read(cls, source: BinaryIO) -> 'SerializableDistribution':
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=True) as tmp:
            tmp.write(source.read())
            return cls.load(tmp.name)
    
    def save(self, target: str | BinaryIO):
        raise NotImplementedError
    
    @classmethod
    def load(cls, source: str | BinaryIO) -> 'SerializableDistribution':
        raise NotImplementedError
    
    def __getstate__(self):
        """Called by jsonpickle during encode."""
        # 1. Create an in-memory binary buffer
        buffer = io.BytesIO()
        # 2. Use your custom write method to dump state into the buffer
        self.write(buffer)
        # 3. Convert binary -> base64 string (so it fits in JSON)
        b64_str = base64.b64encode(buffer.getvalue()).decode('ascii')
        # 4. Return the dict that jsonpickle will actually write
        return {'_byte_stream': b64_str}

    def __setstate__(self, state):
        """Called by jsonpickle during decode."""
        # 1. Decode the base64 string back to binary
        raw_bytes = base64.b64decode(state['_byte_stream'])
        buffer = io.BytesIO(raw_bytes)
        
        # 2. Use the factory method to create the fully loaded NEW instance
        #    Note: self.read() returns a new object, it does not modify self!
        loaded_instance = self.read(buffer)
        
        # 3. TRANSPLANT: Copy the state from the loaded instance to self
        self.__dict__.update(loaded_instance.__dict__)
    
SerializableDistributionT = TypeVar("SerializableDistributionT", bound=SerializableDistribution)