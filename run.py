# run.py

from app.routes import main
from flask import Flask

app = Flask(__name__)
app.register_blueprint(main)

# Установи секретный ключ для работы сессий
app.secret_key = 'your_very_secret_key_here_12345'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)