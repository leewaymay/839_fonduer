
def get_rgx_matcher():

    prefix_rgx = '(\(?((mono|bi|di|tri|tetra|hex|hept|oct|iso|a?cycl|poly).*)?(meth|carb|benz|fluoro|chloro|bromo|iodo|hydro(xy)?|amino|alk).+)'
    suffix_rgx = '(.+(ane|yl|adiene|atriene|kene|k?yne|anol|anediol|anetriol|anone|acid|amine|xide|dine?|(or?mone)|thiol|cine?|rine?|thine?|tone?)s?\)?)'

    dash_rgx = '((\w+\-|\(?)([a-z|\d]\'?\-)\w*)'
    comma_dash_rgx = '((\w+\-|\(?)([a-z|\d]\'?,[a-z|\d]\'?\-)\w*)'
    inorganic_rgx = '(([A-Z][a-z]?\d*\+?){2,})'

    org_rgx = '|'.join([prefix_rgx, suffix_rgx, dash_rgx, comma_dash_rgx, inorganic_rgx])

    return org_rgx