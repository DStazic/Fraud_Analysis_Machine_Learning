#!/usr/bin/python

from collections import defaultdict
import re
import os
from EmailBase import EmailBase

def EmployeeEmails(set1, set2):
    '''
    iterate over email addresses from the main dataset (complete_df) and poi_emails (list of multiple email
    addresses for each poi) datasets and assign each address to the coresponding employee name.
    The function will return a dictionary with email and employee name as key:value pairs.
        
    set1: poi_emails dataset
    set2: main dataset
    '''
    
    # create a key:value pair for listed emails in in the poi_emails dataset (set1)
    #--> key = email address; value = employee name
    email_by_name = dict()
    
    for email in set1:
        for index in set2.index:
            name = index.split()[0].lower()
            if name in email:
                email_by_name[email] = index

    # create a key:value pair for listed emails in the main dataset (set2)
    #--> key = email address; value = employee name
    email_names_df = set2[["email_address"]].dropna().reset_index()
    for index in range(len(email_names_df)):
        email_address = email_names_df["email_address"][index]
        name = email_names_df["index"][index]
        
        if email_address not in email_by_name.keys():
            email_by_name[email_address] = name

    return email_by_name



class EmailPaths(EmailBase):
    '''
    Returns a dictionary with each enron employee present in the main dataset as a key and a list containing all 
    paths to emails that have been written by the cognate employee as the corresponding value.
    Paths to emails that can't be assigned to a specific employee will be assigned to the key "unknown employee".
    
    maildir: root path to the folder containing the email dataset (each employee as a subfolder within root)
    email_set: dataset containing all email addresses associated with an enron employee
    '''
    
    def __init__(self, maildir):
        #self.email_set = email_set
        self.maildir = maildir

    def extractPath(self,email_set):
        email_sender = defaultdict(list)
    
        for paths, subdirs,  files in os.walk(self.maildir):
            for email in files:
                if not email.startswith("."):
                
                    path = os.path.join(paths, email)
                    email_file = open(path)
                    email_text = email_file.read()
                    address = self.ExtractEmailAddress(email_text, "sent")
                    # ExtractEmailAdress returns a list
                    #--> email author is always a single entry in the list; multiple email recepients possible
                    author = self.IdentifyEmailAddress(address[0], email_set, employee = False)
                
                    if author:
                        email_sender[author].append(path)
                    #elif not author:
                    #   email_sender["unknown employee"].append(path)
                    email_file.close()

        return email_sender
                    
    @staticmethod
    def EmployeeEmails(set1, set2):
        '''
        iterate over email addresses from the main dataset (complete_df) and poi_emails (list of multiple email
        addresses for each poi) datasets and assign each address to the coresponding employee name.
        The function will return a dictionary with email and employee name as key:value pairs.
        
        set1: poi_emails dataset
        set2: main dataset
        '''
    
        # create a key:value pair for listed emails in in the poi_emails dataset (set1)
        #--> key = email address; value = employee name
        email_by_name = dict()
    
        for email in set1:
            for index in set2.index:
                name = index.split()[0].lower()
                if name in email:
                    email_by_name[email] = index

        # create a key:value pair for listed emails in the main dataset (set2)
        #--> key = email address; value = employee name
        email_names_df = set2[["email_address"]].dropna().reset_index()
        for index in range(len(email_names_df)):
            email_address = email_names_df["email_address"][index]
            name = email_names_df["index"][index]
        
            if email_address not in email_by_name.keys():
                email_by_name[email_address] = name

        return email_by_name