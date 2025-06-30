import bratiaa as biaa
import re

project = './bratiia_synthetic'

def token_func(text):
    token = re.compile('\w+|[^\w\s]+')
    for match in re.finditer(token, text):
        yield match.start(), match.end()

# token-level agreement
f1_agreement = biaa.compute_f1_agreement(project, token_func=token_func)

# print agreement report to stdout
biaa.iaa_report(f1_agreement)

# agreement per label
label_mean, label_sd = f1_agreement.mean_sd_per_label()

# agreement per document
doc_mean, doc_sd = f1_agreement.mean_sd_per_document() 

# total agreement
total_mean, total_sd = f1_agreement.mean_sd_total()
