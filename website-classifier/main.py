import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
import argparse
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import os
import urllib.request
import languagenet
import dill
import re
import torchtext
import pandas as pd
import collections

parser = argparse.ArgumentParser(description='LanguageNet Classifier')
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
    model_path = 'model.pth'
    model_language_field_path = 'model-language_field.pth'
    model_text_field_path = 'model-text_field.pth'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for file_path in [model_path, model_language_field_path, model_text_field_path]:
        if not os.path.exists(folder_path + '/' + file_path):
            file_url = 'https://media.githubusercontent.com/media/valeriano-manassero/artificial-intelligence-pocs/master/language-detection/saved_model/' + file_path
            with urllib.request.urlopen(file_url) as response, open(folder_path + '/' + file_path, 'wb') as file:
                data = response.read()
                file.write(data)

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

                    df = pd.DataFrame()
                    df = df.append({'text': text_data}, ignore_index=True)
                    df.to_csv(folder_path + '/url.csv', index=False)

                    EMBEDDING_DIM = 32
                    MAX_LENGTH = 10000

                    language_field = torch.load('./' + folder_path + '/' + model_language_field_path, pickle_module=dill)
                    text_field = torch.load('./' + folder_path + '/' + model_text_field_path, pickle_module=dill)
                    tt_df = torchtext.data.TabularDataset(path='data/url.csv',
                                                          format='csv',
                                                          skip_header=True,
                                                          fields=[('text', text_field)])

                    tt_iter = torchtext.data.BucketIterator(dataset=tt_df, batch_size=1, shuffle=False, sort=False)

                    model = languagenet.LanguageNet(len(text_field.vocab), EMBEDDING_DIM, MAX_LENGTH, len(language_field.vocab))
                    model.load_state_dict(torch.load('./' + folder_path + '/' + model_path))

                    data = next(iter(tt_iter))
                    log_probs = model(data.text.t())
                    pred = language_field.vocab.itos[log_probs.max(1)[1]]
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
