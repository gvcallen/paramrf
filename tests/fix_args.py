from pmrf import Model

class BaseModel(Model):
    # b: float = field(default=1.0, static=True)
    a: float

class DerivedModel(BaseModel):
    c: float