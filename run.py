from boostlr_website import app
from boostlr_website.utils import *


from sklearn.ranking.utils import *

creates_dirs()
start_jvm()  # Start JVM when the Flask app starts
if jpype.isJVMStarted():
        System = JClass("java.lang.System")
        classpath = System.getProperty("java.class.path")
        print("JVM Classpath:", classpath)
try:
    app.run(host="0.0.0.0", port=5000, debug=True)
finally:
    import jpype
    if jpype.isJVMStarted():
        jpype.shutdownJVM()  # Shutdown JVM when the app exits

