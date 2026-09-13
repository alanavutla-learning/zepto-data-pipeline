import pandas as pd
def clean_books(book_data):
    df = pd.DataFrame(book_data)
    #df1=pd.DataFrame(book_data)
    #print(df.head())
    #print (df.shape)
    #print(df.columns)
    #df=df[""]
    df["price"]=df['price'].str.strip("Â£")
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
    #print(df['price'])
    df['rating']=df["star_rating"].replace({"One":1,"Two":2,"Three":3,"Four":4,"Five":5})

    #df["in_stock"]=df["availability"].str.strip("").replace("In stock",True)
    #print(df['availability'].sample(30))
    df["in_stock"] = df["availability"].str.contains(
        "In stock",
        case=False,
        na=False
    )
    #print(df["price"].dtype)
    df['price_inr']=df['price']*105.50
    #print(df[["price","price_inr"]])
    #print(df.isnull().sum())
    df = df.rename(columns={'price': 'price_gbp'})
    #print(df.dtypes)

    df['category']=df['category'].str.strip("")
    #print(df['availability'].unique())
   # df = df.rename(columns={'availability': 'in_stock'})
    #df['availability']='In Stock'
    #print(df.describe())
    # Save to your current working directory
    #df.to_csv('data_pipeline/data/processed/books_clean.csv', index=False)
   # print("succes")
    return df