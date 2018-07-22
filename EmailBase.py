import re
from datetime import datetime

class EmailBase(object):
    '''
    Base class for processing email data
    '''
    
    def RemoveMetaData(self,email):
        '''
        Removes meta data from an email (fields denoting sender, recepient, etc.).
        Returns plain email text.
        
        email: email text body as (file object accessed via open() function); string
        '''
    
        # define position within email that indicates the start of the actual text body
        # --> text is preceded by metadata
        #meta_ref = re.compile(r"X-FileName.+\s+")
        meta_ref = re.compile(r"X-FileName.+")
        meta = re.search(meta_ref,email)
        # first item in content list contains metadata, second item contains the text body
        return email.split(meta.group())
    
    
    def EmailSubject(self,email_text):
        '''
        returns email subject
        '''
    
        regex_ref = re.compile(r"(?<=Subject:).*(?=\n)")
        match = re.search(regex_ref, email_text)
    
        return match.group()


    def EmailID(self,email_text):
        '''
        creates unique email ID from timestamp subject and last substring in text body
        '''
    
        regex_ref = re.compile(r"\S+$")
        match = re.search(regex_ref, email_text)
        if match:
            match = match.group()
        else:
            match = "whitespace"
    
        timestamp = self.EmailTimestamp(email_text)
        subject = self.EmailSubject(email_text)
        email_id = match + str(timestamp) + subject
    
        return email_id


    def EmailTimestamp(self,email_text):
        '''
        returns email timestamp
        '''
    
        # match following timestamp format: 12 Dec 2000 12:19:00
        regex_ref = re.compile(r"(?<=Date:\s\w{3},\s)\d{1,2}\s\w{3}\s\d{4}\s\d{1,2}:\d{2}:\d{2}")
        match = re.search(regex_ref, email_text)
        #print match
        timestamp = datetime.strptime(match.group(), "%d %b %Y %H:%M:%S")
    
        return timestamp


    def ClassAffiliation(self,name, dataset):
        '''
        checks class affiliation (poi versus non-poi) for a name corresponding to from and sent email address, respectively.
        returns class if name matches employee name from the main dataset, otherwise returns false.
        '''
    
        try:
            is_poi = dataset["poi"].loc[name]
            return is_poi
        except KeyError:
            return False


    def ExtractEmailAddress(self,email, identifier="sent"):
        '''
        given an email as input, returns email address from sender or recepient
        
        file_in: file object (accessed via open() function)
        identifier: specify email address to be extracted from:
        1.sender (identifier="sent") or 2.recepient (identifier="received")
        '''
    
        # extract meta data from email text body; avoids regex mismatches if text body also contains
        # substrings that resemble email meta data
        email_text = self.RemoveMetaData(email)[0]
    
        meta_ref_from = re.compile(r"(?<=From:).+(?=\s)")
        match_from = re.search(meta_ref_from, email_text)
    
        meta_ref_to = re.compile(r"(?<=To:)(.|\n)+(?=Subject:)")
        match_to = re.search(meta_ref_to, email_text)
    
        meta_ref_cc = re.compile(r"(?<=Cc:)(.|\n)+(?=Mime-Version:)")
        match_cc = re.search(meta_ref_cc, email_text)
    
        # specify what type of email to extract
        if identifier == "sent":
            if match_from:
                return re.sub(r"(\n|\s)","",match_from.group()).split(",")

        if identifier == "received":
            # verify that email has receiver (to or to/cc); some emails have not a valid receiver field!
            if match_to:
                email_to = re.sub(r"(\n|\s)","",match_to.group()).split(",")
                return email_to

        if identifier == "shared":
            if match_cc:
                return re.sub(r"(\n|\s)","",match_cc.group()).split(",")
    
        # if no email address present
        return False


    def IdentifyEmailAddress(self,address, email_set, employee = True):
        '''
        checks whether an email address belongs to an enron employee from the dataset.
        If true, employee name is returned.
        
        address: extracted sender email address
        email_set: dataset containing all email addresses associated with an enron employee
        '''
    
        try:
            from_employee = email_set[address]
            return from_employee
    
        except KeyError:
            if address:
                return "unknown employee"
        return False
