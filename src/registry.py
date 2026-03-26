from models.stackdili.model import Model as STACKDILI
from models.my_model.model import Model as MY_MODEL

MODEL_REGISTRY = {
    "stackdili": STACKDILI,
    "my_model": MY_MODEL,
}