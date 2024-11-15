from flask import Flask

app = Flask(__name__)

from boostlr_website import routes
