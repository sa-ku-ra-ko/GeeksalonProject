from flask import (
     Flask, 
     request, 
     render_template)

from model import predict

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_user_text():
    # フォームからテキストを受け取る
    user_text = request.form['user_text']

    # model.py の predict 関数を呼び出す
    result = predict(user_text)

    # 結果を result.html に渡して表示
    return render_template(
        'result.html',
        title=result['title'],
        author=result['author'],
        summary=result['summary'],
        image_url=result['image_url'],
        input_text=result['input_text']
    )

if __name__ == "__main__":
    app.run(debug=True)