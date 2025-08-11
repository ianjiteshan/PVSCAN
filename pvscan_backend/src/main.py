import os
import sys
from flask import Flask, send_from_directory, render_template
from flask_cors import CORS
from src.models.user import db
from src.routes.user import user_bp
from src.routes.analysis import analysis_bp

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Paths for static and template folders
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, 'static')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, static_folder='static', template_folder='templates')

app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT'

# Enable CORS
CORS(app, origins="*")

# Register blueprints
app.register_blueprint(user_bp, url_prefix='/api')
app.register_blueprint(analysis_bp, url_prefix='/api')

# Configure uploads and database
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(BASE_DIR, 'database', 'app.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
with app.app_context():
    db.create_all()

# Serve frontend
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(os.path.join(STATIC_DIR, path)):
        return send_from_directory(STATIC_DIR, path)
    else:
        return render_template("index.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
