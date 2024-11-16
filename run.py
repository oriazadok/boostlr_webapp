from boostlr_website import app
from boostlr_website.src.utils import *
from boostlr_website.utils import *

creates_dirs()
start_jvm()  # Start JVM when the Flask app starts
try:
    app.run(host="0.0.0.0", port=5000, debug=True)
finally:
    import jpype
    if jpype.isJVMStarted():
        jpype.shutdownJVM()  # Shutdown JVM when the app exits

