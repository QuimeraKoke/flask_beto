import os.path
from flask import Flask, jsonify, request, Response
import traceback

from bert_model import BETO

app = Flask(__name__)

PORT_NUMBER = 8080
beto = BETO()


def root_dir():  # pragma: no cover
    return os.path.abspath(os.path.dirname(__file__))


def get_file(filename):  # pragma: no cover
    try:
        src = os.path.join(root_dir(), filename)
        return open(src).read()
    except IOError as exc:
        return str(exc)


@app.route('/', methods=["POST"])
def predict():
    print(request.form['diagnostic'])
    diagnostic = request.form['diagnostic']
    tmp = beto.infer(diagnostic)
    resp = []
    for r in tmp:
        a, b = r
        resp.append({'name': a, 'percentage': b})
    return jsonify({'predictions': resp})


@app.route('/', methods=["GET"])
def index():
    content = get_file('index.html')
    return Response(content, mimetype="text/html")


@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify(stackTrace=traceback.format_exc())


if __name__ == '__main__':
    app.run(port=PORT_NUMBER)
