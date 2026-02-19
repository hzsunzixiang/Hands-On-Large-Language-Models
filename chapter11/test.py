from datasets import load_dataset

import ssl
import os
os.environ['CURL_CA_BUNDLE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context
# 使用 trust_remote_code 和重试
dataset = load_dataset("conll2003", trust_remote_code=True)
