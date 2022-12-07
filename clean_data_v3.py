import pandas as pd
import string
import re


class clean_data:
    def __init__(self, file_path):
        self.java_keywords = ["abstract", "continue", "for", "new", "switch", "assert", "default", "goto", "package",
                              "synchronized",
                              "boolean", "do", "if", "private", "this", "break", "double", "implements", "protected",
                              "throw",
                              "byte", "else", "elseif", "import", "public", "throws", "case", "enum", "instanceof",
                              "return",
                              "transient",
                              "catch", "extends", "int", "short", "try",
                              "char", "final", "interface", "static", "void",
                              "class", "finally", "long", "strictfp", "volatile",
                              "const", "float", "native", "super", "while"]
        self.path = file_path
        self.df = pd.read_csv(self.path, sep='@!@', header=None, names=['class', 'method', 'detail'],
                              engine='python')
        self.class_detail = self.df[self.df['method'].str.contains(pat=r'^(?:test)\w+|^setUp')]

    def clean(self):
        data = self.class_detail['detail']
        # split word with underscore
        clean_data = data.apply(lambda s: ' '.join(s.split('_')))
        # delete white space
        clean_data = clean_data.apply(
            lambda s: s.translate(str.maketrans(string.whitespace, ' ' * len(string.whitespace), '')))
        # split word with Capital letter
        clean_data = clean_data.apply(lambda s: ' '.join(re.sub(r"([A-Z])", r" \1", s).split()))
        # split word with numerical by Tulyakai approach
        clean_data = clean_data.apply(lambda s: ' '.join(re.split('(-?\d+\.?\d*)', s)))
        clean_data = clean_data.apply(lambda s: ''.join(i for i in s if not i.isdigit()))
        # remove special characters
        clean_data = clean_data.apply(
            lambda s: s.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).replace(' ' * 4,
                                                                                                            ' ').replace
            (' ' * 3, ' ').replace(' ' * 2, ' ').strip())
        # cast to lower
        clean_data = clean_data.apply(lambda s: s.lower())
        # split word with java keyword
        clean_data = clean_data.apply(lambda s: ''.join(re.sub(r"\b(%s)\b" % "|".join(self.java_keywords), "", s)))
        # delete whitespace
        clean_data = clean_data.apply(lambda s: ' '.join(w.strip() for w in s.split()))
        clean_data = pd.concat([self.class_detail['class'], self.class_detail['method'], clean_data], axis=1)

        clean_data.to_pickle("resource/v_3_of_Java_extraction/CleanDetailDataframe.pkl")


if __name__ == '__main__':
    s = clean_data('resource/v_3_of_Java_extraction/ClassAndMethod.txt')
    s.clean()
