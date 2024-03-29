import logging

ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)

bentoml_logger = logging.getLogger("bentoml")
bentoml_logger.addHandler(ch)
# Try to change this to ERROR or others, remember log level order
# DEBUG -> INFO -> WARN -> ERROR -> CRITICAL
bentoml_logger.setLevel(logging.INFO)
