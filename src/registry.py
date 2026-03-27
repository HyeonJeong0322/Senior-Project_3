from models.stackdili.model import Model as STACKDILI
from models.stackdili_fixed.model import Model as STACKDILI_FIXED
from models.my_model.model import Model as MY_MODEL

MODEL_REGISTRY = {
    "stackdili": STACKDILI,
    "my_model": MY_MODEL,
    "stackdili_fixed": STACKDILI_FIXED
}