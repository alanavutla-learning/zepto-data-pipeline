import requests
#import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
def scrape_books():
    url = "https://books.toscrape.com/"
    response = requests.get(url)
    soup=BeautifulSoup(response.text,"html.parser")
    #below code line by line add 
    #print(response.status_code)#if 200 connection successful and response has entire ebpage and we can download it
    #print(response.text[:1000])#to check the webpage content1000 lines 
    #print(soup.title)#so that we can parse it and look for required content
    #print(soup.select(".side_categories"))#The website's HTML gives the Categories section a CSS class called:side_categories
    #. means:"Find an HTML element whose class is side_categories."
    category_links = soup.select(".side_categories ul li ul li a")

    '''for link in category_links:
        category_name = link.get_text(strip=True)
        category_url = link.get("href")
    # print(category_name, "→", category_url)
    category_url = "https://books.toscrape.com/catalogue/category/books/travel_2/index.html"

    category_response = requests.get(category_url)

    category_soup = BeautifulSoup(category_response.text, "html.parser")


    #print(category_response.status_code)
    #print(category_soup.title)
    first_book = category_soup.select("article.product_pod")[0]
    print("Number of books:", len(books))
    #print(first_book)#to get first book details and class names required to extract our required fields
    #GETTING TITLE OF FIRST BOOK
    title = first_book.select_one("h3 a").get("title")
    print("Title:", title)
    #GETTING price OF FIRST BOOK whose class named in details as price color

    price = first_book.select_one(".price_color").get_text(strip=True)
    print("Price:", price)
    rating_element = first_book.select_one(".star-rating") 
    star_rating = rating_element.get("class")[1]
    print("Star rating:", star_rating)

    availability = first_book.select_one(".availability").get_text(strip=True) 
    print("Availability:", availability)'''

    '''category_name = "Travel"
    books = category_soup.select("article.product_pod")'''

    '''for book in books: 
        title = book.select_one("h3 a").get("title") 
        price = book.select_one(".price_color").get_text(strip=True)
        rating_element = book.select_one(".star-rating") 
        star_rating = rating_element.get("class")[1] 
        availability = book.select_one(".availability").get_text(strip=True)
        print(title) 
        print(price) 
        print(star_rating) 
        print(availability) 
        print(category_name)
        print("--------------------")


    selected_categories = category_links[:3]

    for link in selected_categories:
        category_name = link.get_text(strip=True)
        category_url = "https://books.toscrape.com/" + link.get("href")

        #print("CATEGORY:", category_name)
        #print("URL:", category_url)

        category_response = requests.get(category_url)
        category_soup = BeautifulSoup(category_response.text, "html.parser")

        books = category_soup.select("article.product_pod")

        #print("Number of books:", len(books))
        #print("--------------------")


    next_link = category_soup.select_one("li.next a")

    if next_link:
        print("Next page:", next_link.get("href"))
    else:
        print("No next page")'''

    book_data = []

    selected_categories = category_links[:3]

    for link in selected_categories:

        category_name = link.get_text(strip=True)
        current_url = urljoin(url, link.get("href"))

        #print("CATEGORY:", category_name)

        while current_url:

            #print("Scraping:", current_url)

            response = requests.get(current_url)
            soup = BeautifulSoup(response.text, "html.parser")

            books = soup.select("article.product_pod")

            #print("Books on this page:", len(books))

            for book in books:
                title = book.select_one("h3 a").get("title")
                price = book.select_one(".price_color").get_text(strip=True)

                rating_element = book.select_one(".star-rating")
                star_rating = rating_element.get("class")[1]

                availability = book.select_one(".availability").get_text(strip=True)

                book_record = {
                    "title": title,
                    "price": price,
                    "star_rating": star_rating,
                    "availability": availability,
                    "category": category_name
                }

                book_data.append(book_record)

            next_link = soup.select_one("li.next a")

            if next_link:
                current_url = urljoin(current_url, next_link.get("href"))
            else:
                current_url = None

    #print("Total records:", len(book_data))
    #print(book_data[0])
    return book_data


'''book_data=scrape_books()
df_raw = pd.DataFrame(book_data)
df_raw.to_csv(
    "data_pipeline/data/raw/books_raw.csv",
    index=False
)'''