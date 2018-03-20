library("elastic")
connect()  
docs_bulk(inputSubsetDF,doc_ids = inputSubsetDF$Dataset_Index,index="DNN")