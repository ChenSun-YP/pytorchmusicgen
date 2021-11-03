
import data



if __name__ =="__main__":
  
  dataset = data.get_gtzan(subset='validation')
  print(dataset[180])