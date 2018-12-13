import json
import re

dir                 = '/Users/8kd/workdir/data/cancer-reports/';
filename            = 'matched_fd.json';
valid_task_labels   = [ 'gs_organ_label', 
                        'gs_icd_label', 
                        'gs_lat_label', 
                        'gs_behavior_label',
                        'gs_hist_grade_label'
                        ];

def read_json(file):
    """
    function to read report as list
    """
    with open(file) as data_file:
        data = json.load(data_file)
    return data

def cleanText(text):
    '''
    function to clean text
    '''
    #replace symbols and tokens
    text = re.sub('\n|\r', ' ', text)
    text = re.sub('o clock', 'oclock', text, flags=re.IGNORECASE)
    text = re.sub(r'(p\.?m\.?)','pm', text, flags=re.IGNORECASE)
    text = re.sub(r'(a\.?m\.?)', 'am', text, flags=re.IGNORECASE)
    text = re.sub(r'(dr\.)', 'dr', text, flags=re.IGNORECASE)
    text = re.sub('\*\*NAME.*[^\]]\]', 'nametoken', text)
    text = re.sub('\*\*DATE.*[^\]]\]', 'datetoken', text)
    text = re.sub("\?|'", '', text)
    text = re.sub('[^\w.;:]|_|-', ' ', text)
    text = re.sub('[0-9]+\.[0-9]+','floattoken', text)
    text = re.sub('floattokencm','floattoken cm', text)
    text = re.sub(' [0-9][0-9][0-9]+ ',' largeint ', text)
    text = re.sub('\.', ' . ', text)
    text = re.sub(':', ' : ', text)
    text = re.sub(';', ' ; ', text)

    #lowercase
    text = text.lower()

    #tokenize
    text = text.split()
    
    # remove smaller token
    text = [x for x in text if len(x)>2]
    
    return text

def get_valid_reports_label(reports_json):
    """
    function to get text, labels for valid tasks
    """
    valid_entries = [x for x in reports_json 
       if x['gs_organ_label']['match_status']=="matched"
       and x['gs_icd_label']['match_status']=="matched"
       and x['gs_lat_label']['match_status']=="matched"
       and x['gs_behavior_label']['match_status']=="matched"
       and x['gs_hist_grade_label']['match_status']=="matched"]
    
    valid_labels = dict();
    
    for task_name in valid_task_labels:
        valid_labels[task_name] = [ x[task_name]['match_label'] 
            for x in valid_entries ]
    
    valid_text = [x['doc_raw_text'] for x in valid_entries]
    
    valid_tokens = [cleanText(x) for x in valid_text]
    
    return list(zip(valid_tokens, valid_labels['gs_organ_label'], 
                    valid_labels['gs_icd_label'], 
                    valid_labels['gs_lat_label'],
                    valid_labels['gs_behavior_label'],
                    valid_labels['gs_hist_grade_label']))

def main(args = None):
    reports_json = read_json(dir + filename);
    reports_tuple_with_labels = get_valid_reports_label(reports_json);
    
    for report in reports_tuple_with_labels:
        print('\n')
        print(report)
        print('\n')
        input("Press Enter to continue...")
    
    print('-------------done------------')
    

if __name__ == '__main__':
    main()

"""
Created on Wed Dec 12 10:37:17 2018

@author: 8kd

based on code obtained from Hong-Jun
"""
