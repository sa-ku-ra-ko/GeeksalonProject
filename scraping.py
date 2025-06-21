import requests
from bs4 import BeautifulSoup
import csv
import time


# 書籍詳細ページからデータを抽出
def scrape_book_from_detail_page(book_id):
    url = f"https://bookmeter.com/books/{book_id}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.select_one("h1.inner__title").text.strip()
        author = soup.select_one("ul.header__authors").text.strip()
        image_tag = soup.select_one("a.image__cover img")
        image_url = image_tag["src"] if image_tag else "画像なし"
        summary_tag = soup.select_one("div.book-summary__default")
        summary = (
            summary_tag.get_text(separator="\n").strip()
            if summary_tag
            else "あらすじ情報なし"
        )

        return {
            "タイトル": title,
            "著者": author,
            "リンク": url,
            "画像URL": image_url,
            "あらすじ": summary,
        }

    except Exception as e:
        print(f"[{book_id}] エラー: {e}")
        return None


# 書籍リストをCSVに保存
def save_books_to_csv(books, filename="bookmeter_books.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["タイトル", "著者", "リンク", "画像URL", "あらすじ"],
        )

        writer.writeheader()
        writer.writerows(books)


# 実行：IDを自動生成して取得
if __name__ == "__main__":
    start_id = 100000  # ← 任意に調整可
    end_id = 200000  # ← 任意に調整可（例：1～30000）
    max_books = 10000  # ← 取得する最大件数（任意）

    books = []
    for book_id in range(start_id, end_id + 1):
        print(f"[INFO] 書籍ID {book_id} を取得中...")
        book = scrape_book_from_detail_page(book_id)
        if book:
            books.append(book)

        if len(books) >= max_books:
            print(f"[STOP] 最大 {max_books} 件に達したため終了")
            break

        time.sleep(2)  # サーバー負荷を避けるため

    save_books_to_csv(books)
    print(f"[DONE] {len(books)} 件を 'bookmeter_books.csv' に保存しました。")