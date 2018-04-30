import os
ATTRIBUTE = "organic_figure"

def set_up(name):
    if name == 'wei':
        return set_up_wei()
    if name == 'zhewen':
        return set_up_zhewen()
    if name == 'xiuyuan':
        return set_up_xiuyuan()

def set_up_wei():
    os.environ['FONDUERHOME'] = '/Users/liwei/BoxSync/s2016/Dropbox/839_fonduer'
    os.environ['FONDUERDBNAME'] = ATTRIBUTE
    os.environ['SNORKELDB'] = 'postgres://localhost:5432/' + os.environ['FONDUERDBNAME']
    docs_path = os.environ['FONDUERHOME'] + '/tutorials/organic_synthesis_figures/data/html/'
    pdf_path = os.environ['FONDUERHOME'] + '/tutorials/organic_synthesis_figures/data/pdf/'
    return docs_path, pdf_path

def set_up_zhewen():
    os.environ['FONDUERHOME'] = '/Users/liwei/BoxSync/s2016/Dropbox/839_fonduer'
    os.environ['FONDUERDBNAME'] = ATTRIBUTE
    os.environ['SNORKELDB'] = 'postgres://localhost:5432/' + os.environ['FONDUERDBNAME']
    docs_path = os.environ['FONDUERHOME'] + '/tutorials/organic_synthesis_figures/data/html/'
    pdf_path = os.environ['FONDUERHOME'] + '/tutorials/organic_synthesis_figures/data/pdf/'
    return docs_path, pdf_path

def set_up_xiuyuan():
    os.environ['FONDUERHOME'] = '/Users/liwei/BoxSync/s2016/Dropbox/839_fonduer'
    os.environ['FONDUERDBNAME'] = ATTRIBUTE
    os.environ['SNORKELDB'] = 'postgres://localhost:5432/' + os.environ['FONDUERDBNAME']
    docs_path = os.environ['FONDUERHOME'] + '/tutorials/organic_synthesis_figures/data/html/'
    pdf_path = os.environ['FONDUERHOME'] + '/tutorials/organic_synthesis_figures/data/pdf/'
    return docs_path, pdf_path