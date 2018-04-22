def news_text_clean(data):
    '''
    Function to clean news from coindesk 
    '''
    import re
    
    data['title'] = data['title'].apply(lambda x: x.replace('&amp',' and ')) # This appears in place of & due to encoding issue
    data['contents'] = data['contents'].apply(lambda x: x.replace('&amp',' and '))

    data['title'] = data['title'].apply(lambda x: x.replace("'","")) #Replace '. Other symbols will be ignored by keras tokenizer
    data['contents'] = data['contents'].apply(lambda x: x.replace("'",""))

    data['title'] = data['title'].apply(lambda x: re.sub(r'\d+', '', x)) # Remove numbers
    data['contents'] = data['contents'].apply(lambda x: re.sub(r'\d+', '', x))

    data['title'] = data['title'].apply(lambda x: x.replace('–','')) # Remove special symbols
    data['contents'] = data['contents'].apply(lambda x: x.replace('–',''))
    
    # Remove \xa0
    data['contents'] = data['contents'].apply(lambda x: x.replace(u'\xa0', u' '))

    # Remove text after "The leader in blockchain news" for CoinDesk articles
    data['contents'] = data['contents'].apply(lambda x: x.split('The leader in blockchain news')[0])
    data['contents'] = data['contents'].apply(lambda x: x.split('Disclosure:')[0])
    data['contents'] = data['contents'].apply(lambda x: x.split('Disclaimer:')[0])

    
    return data