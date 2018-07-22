from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from EmailBase import EmailBase
import re
import string
import os
import pandas as pd

class EmailStemmer(EmailBase):
    
    def __init__(self,email_paths,dataset):
        self.email_paths = email_paths
        self.dataset = dataset
    
    def wordStemmer(self):
        '''
        returns initialized SnowballStemmer class
        '''
        return SnowballStemmer("english")
    
    def NamesReferenceNLP(self):
        
        '''
        Extracts first and last names from different csv files that contain names of male/female
        caucasian and african american origin and applies stemming. returns a set of stemmed first and last names.
        '''
        
        names_set = set()
        sex = ["Male", "Female"]
        ethnicity = ["White", "Black"]
        
        
        for eth in ethnicity:
            for sx in sex:
                # define path for each csv file;
                file_in = "{0}_{1}_Names.csv".format(eth, sx)
                path = os.path.join(os.getcwd(),"NLP_names_dataset", file_in)
                
                # read csv file as dataframe and update columns (some columns seem to have whitespace)
                df = pd.read_csv(path)
                df.columns = pd.Series(df.columns).apply(lambda col: col.strip()).tolist()
                
                # extract first and last name (middle names as one letter abreviations; ignore those)
                first_name = df["first name"].apply(lambda name: name.split()[0])
                last_name = df["last name"]
                names = pd.concat([first_name, last_name])
                
                # populate set
                for name in names:
                    names_set.add(self.wordStemmer().stem(name))
    
        return names_set



    def EmployeeNames(self):
        '''
        Extracts the stemmed variant of employee first and second names from the main dataset (data_df).
        This set will be used as a reference to remove employee names from an email.
        Returns a set of stemmed employee first and second names
        
        dataset: main dataset; dataframe
        '''
            
        names = set()
                
        for name in self.dataset.index:
            name = name.split()
            # sort split name according to string length and pick only the two largest strings
            #--> makes sure that middle name and/or generation titles are omitted
            name = sorted(name, key = len, reverse = True)
            names.add(self.wordStemmer().stem(name[0]))
            names.add(self.wordStemmer().stem(name[1]))
                                    
        return names


    def GetParagraphs(self, email):
        '''
        returns all paragraphs present in an email (paragraphs preceeded and followed by whitespace)
        '''
            
        # condense excessive next line-type of whitespace in order to avoid RuntimeError: maximum recursion
        # in the downstream recursive approach to clean the email
        streamlined_text = re.sub(r"\n{3,}", "\n\n", email)
                
        # extract paragraphs from email text body
        regex_ref = re.compile(r"\n{2,}")
        match = re.search(regex_ref, streamlined_text)
        if match:
            return streamlined_text.split(match.group())
        else:
            return None


    def LineStart(self,text):
        '''
        checks if line starts with a word-character substring followed by a colon.
        If true, returns the substring (including the colon), if false returns false
        
        text: paragraph from email text body
        '''
            
        regex_ref = re.compile(r"^\w+:?")
        match = re.search(regex_ref, text)
        if match:
            return match.group()
        else:
            return False


    def CleanEmailConversation(self, email, split = False, paragraphs = None, cleaned_text = ""):
    
        '''
        In a recursive approach checks for identifier ("From:", "To:", "cc:", "Subject:") in an email conversation
        (multiple email replies) and removes identifier and the strings they are referencing to.
        Returns native (if email not part of a conversation set) or cleaned (email part of a conversation set) email version.
        
        email: email text body; string
        split: references if email body was split in paragraphs; boolean
        paragraphs: paragraphs present in an email; None or list of strings
        cleaned_text: cleaned email version; string
        '''
        # substrings at the start of a new line that indicate meta data
        identifier = set(['to:', 'cc:', 'bcc:', 'from:', 'subject:', 'sent:', 'date:'])
                
        # find all paragraphs for an email
        if not split:
            paragraphs = self.GetParagraphs(email)
            split = True
                            
        # if no paragraphs present, return native email text
        if paragraphs == None:
            return email
                                    
        # if all paragraphs processed, return cleaned email
        if len(paragraphs) == 0:
            return cleaned_text
                                            
        # recursive approach to check and remove remaining meta data; cleaning will be conducted for each paragraph
        else:
            paragraph = paragraphs.pop(0)
            line_start = self.LineStart(paragraph)
            found_identifier = set(re.findall(r"[aA-zZ]+:", paragraph))
            identifier_intersection = identifier.intersection(found_identifier)
                                                                
            # meta data paragraph if at least 2 meta identifier are present; continue with next paragraph
            #--> assumption is that two identifier are a very likely indicator of meta data
            if len(identifier_intersection) >= 2:
                return self.CleanEmailConversation(cleaned_text, split, paragraphs, cleaned_text)
                                                                        
            # if only 1 meta identifier is present, check if line starts with respective identifier; continue with next paragraph
            #--> a one line meta paragraph is assumed if only one identifier is found and the paragraph starts with the identifier
            elif len(identifier_intersection) == 1 and line_start in identifier:
                return self.CleanEmailConversation(cleaned_text, split, paragraphs, cleaned_text)
                                                                                
            # keep non meta data paragraph; add to cleaned_text string
            else:
                cleaned_text = " ".join([cleaned_text, paragraph])
                return self.CleanEmailConversation(cleaned_text, split, paragraphs, cleaned_text)


    def CleanForwardedEmail(self, email):
        '''
        Checks if email was forwarded and, when necessary, removes the string that depicts a forwarded email from the
        email body. Also removes "-----Original Message-----" substrings that can be present within the text body of
        an email conversation.
        Returns native (if not forwarded email) or cleaned (if forwarded email) email version.
        
        email: email text body; string
        '''
            
        # search for following type of substring:
        #--> ---------------------- Forwarded by David W Delainey/HOU/ECT on 12/12/2000 12:30 PM ---------------------------
        # look for at least 5 dash literals in order to prevent removal of email text passages that use dash
        # use negative look ahead to prevent regex to encompass first and last target substring if multiple
        # target substrings are present. This would remove any text in between
        # Example string, "-----Original Message----- TEXT -----Original Message-----"
        #--> without negative look ahead the entire string would be flaged for removal
        #--> with negative look ahead only the two substrings "-----Original Message-----" would be flaged for removal and
        #    the "TEXT" substring would remain intact!
        meta_ref = re.compile(r"-{5,}\s?(\bForwarded\b|\bOriginal\b)((?!-{5,})[\w\W])+-{5,}")
        #meta_ref = re.compile(r"-{5,}((?!-{5,})[\w\d\W])+-{5,}")
        return re.sub(meta_ref,"", email)



    def CleanWhitespace(self,email):
        '''
        Replaces all whitespace literals (tab, new line, etc.) with space literal, which helps to remove names from
        emails after stemming.
        Returns native (if no whitespace literals) or cleaned (if whitespace literals) email version.
        
        email: email text body as; string
        '''
            
        re_ref = re.compile(r"\s+")
        return re.sub(re_ref," ",email)


    def ParseText(self, email_text):
        '''
        Takes raw email text, applies cleaning procedures and returns stemmed words.
        
        email_text: email text body; string
        ref_names: stemmed variants of reference names; set (used to remove names prior to tf-idf feature engineering)
        path: single file path referencing an email (authored by an enron employee); string
        
        '''
            
        content = self.RemoveMetaData(email_text) # method from EmailBase class
        words = ""
                    
        # check if the email text body actually contains text after removal of meta data
        if len(content) > 1:
            content = self.CleanForwardedEmail(content[1])
            #try:
            content = self.CleanEmailConversation(content.lower())
            #except RuntimeError:
                #return False
            content = self.CleanWhitespace(content)
            
            # remove special abreviations; if removing those abbr., remove before eliminating
            # punctuation/white space in order to discriminate from other valid words
            #--> removing punctuation/white space: i'll -> ill; not possible to discriminate from the word ill
            abreviations = ["i'll", "i've", "i'd", "i'm", "you'd", "you'll", "you're", "you've", "it'd", "it'll",
                                                                "we're", "we've", "we'll", "we'd"]
            for word in content.split():
                if word.lower() in abreviations:
                    content = content.replace(word,"")
                                                                
            # remove punctuation from text
            text_string = content.translate(string.maketrans("",""), string.punctuation)
            stop_words = stopwords.words("english")
            filtered_text = [word for word in text_string.split() if not word in stop_words]
                                                                            
            # transform words in text to root words by stemming or lemmatizing
            employee_names = self.EmployeeNames()
            ref_names = self.NamesReferenceNLP()
                                                                                        
            normalized_text = []
            for word in filtered_text:
                word_normalized = self.wordStemmer().stem(word)
                if (word_normalized in employee_names) or (word_normalized in ref_names):
                    continue
                normalized_text.append(word_normalized)
                                                                                                                
            words = " ".join(normalized_text)
            return words
        return False
                                                                                                                    
    def ProcessEmail(self):
        '''
        Feeds an email into the cleaning and stemming pipeline. Emails are referenced via a file path. 
        
        email_paths: absolute file paths referencing each email [list item] written by an enron employee present in 
        the main dataset; dictionary
        --> {'ALLEN PHILLIP K':[path1, path2,...]
        
        ref_names: stemmed variants of reference names; set             
        '''
            
        from_data = []
        word_data = []
        unique_emails = set()
                        
        for employee in self.email_paths.keys():
            # exclude emails from unknown authors; only relevant for quantifying email input/output
            if employee != "unknown employee":
                for path in self.email_paths[employee]:
                                    
                    email = open(path, "r")
                    email_text = email.read()
                    email_id = self.EmailID(email_text)
                    email.close()
                                                    
                    # check for duplicate emails
                    if email_id not in unique_emails:
                        unique_emails.add(email_id)
                        processed_text = self.ParseText(email_text)
                                                                
                    # check for text body in emails
                    if processed_text:
                        word_data.append(processed_text)
                        from_data.append(employee)
                                                                            
        return word_data, from_data


