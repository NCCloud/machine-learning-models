import os
import argparse
import re
import collections
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
import fastText


parser = argparse.ArgumentParser(description='FastTextNet Classifier')
parser.add_argument('-u', '--url-file', default='', type=str, help='file containing urls')


def visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def main():
    args = parser.parse_args()
    if args.url_file == '':
        print('No url file specified, exiting...')
        return

    folder_path = 'data'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    cnt = collections.Counter()

    with open(args.url_file, 'r') as url_list:
        for url in url_list:
            try:
                page = requests.get(url.strip(), timeout=30)
                if page.status_code == 200:
                    html_code = page.text
                    soup = BeautifulSoup(html_code, 'html.parser')
                    data = soup.findAll(text=True)
                    result = filter(visible, data)
                    text_data = u" ".join(t.strip() for t in result)
                    text_data = ' '.join(text_data.split())
                    text_data = text_data.replace('\r', '').replace('\n', '')

                    text_data = re.sub(r'[\W_]+', ' ', text_data.lower(), flags=re.IGNORECASE | re.UNICODE)
                    text_data = re.sub("\d+", "", text_data)
                    text_data = re.sub(' +', ' ', text_data)

                    model = fastText.load_model('./lid.176.ftz')
                    pred = model.predict(text_data, 1)[0][0].replace('__label__', '')
                    cnt[pred] += 1
                    with open(folder_path + '/results', 'w') as results:
                        for l in cnt.most_common():
                            results.write(l[0])
                            results.write(':')
                            results.write(str(l[1]))
                            results.write('\n')
                else:
                    print(url, 'ERR:', page.status_code)
            except Exception as ex:
                print(url, 'EXCEPTION:', type(ex).__name__)


if __name__ == '__main__':
    main()
