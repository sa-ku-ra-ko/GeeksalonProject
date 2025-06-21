import requests
from bs4 import BeautifulSoup
import csv
import time
import os

# 対象URL一覧（任意に追加・削除可能）
urls = [
    "https://bookmeter.com/rankings/latest/read_book/bunko",
    "https://bookmeter.com/rankings/latest/wish_book/bunko",
    "https://bookmeter.com/rankings/latest/read_book/tankoubon",
    "https://bookmeter.com/rankings/latest/wish_book/tankoubon",
]

BASE_URL = "https://bookmeter.com"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def scrape_ranking(url):
    books = []
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        items = soup.select("li.list__book")
    except Exception as e:
        print(f"[ERROR] ランキングページ取得失敗: {e}")
        return books

    for book in items[:20]:
        try:
            title_tag = book.select_one("div.detail__title > a")
            title = title_tag.text.strip()
            link = BASE_URL + title_tag["href"]

            author_tag = book.select_one("ul.detail__authors li a")
            author = author_tag.text.strip() if author_tag else "著者不明"

            image_tag = book.select_one("div.book__cover img")
            image_url = image_tag["src"] if image_tag else ""

            # 詳細ページからあらすじ取得
            summary = "あらすじ不明"
            try:
                detail_resp = requests.get(link, headers=HEADERS)
                detail_resp.raise_for_status()
                detail_soup = BeautifulSoup(detail_resp.text, "html.parser")
                summary_tag = detail_soup.select_one("div.book-summary__default")
                summary = summary_tag.text.strip() if summary_tag else "あらすじ不明"
            except Exception as e:
                print(f"[WARN] あらすじ取得失敗: {e}")

            books.append(
                {
                    "タイトル": title,
                    "著者": author,
                    "リンク": link,
                    "画像URL": image_url,
                    "あらすじ": summary,
                }
            )

            time.sleep(1)

        except Exception as e:
            print(f"[ERROR] 書籍処理失敗: {e}")
            continue

    return books


def save_to_csv(books, filename="data/bookmeter_ranking_no_genre.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["タイトル", "著者", "リンク", "画像URL", "あらすじ"]
        )
        writer.writeheader()
        writer.writerows(books)


# メイン実行
if __name__ == "__main__":
    all_books = []
    for url in urls:
        print(f"[INFO] ランキング取得中: {url}")
        books = scrape_ranking(url)
        all_books.extend(books)

    save_to_csv(all_books)
    print(f"[DONE] 合計 {len(all_books)} 件の書籍をCSVに保存しました。")