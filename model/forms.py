from django import forms

class YourFileUploadForm(forms.Form):
    # setFile = forms.FileField(label='Drop .set file', widget=forms.ClearableFileInput(attrs={'accept': '.set'}))
    csvFile = forms.FileField(label='Drop .csv file', widget=forms.ClearableFileInput(attrs={'accept': '.csv'}))
