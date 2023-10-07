library(readxl)
library(tidyverse)
library(ggmap)
#register_google(key = "XXX", write = TRUE) # inserir sua chave

data = read_csv("../web_scraping/cep_df.csv")  %>% 
  mutate(lat2 = NA,
         long2 = NA)

dataf = data


j=1

longlat=matrix(NA,dim(dataf)[1],2)

Tm=dim(data)[1]
for (j in 1:Tm){
  end=paste('cep',data[j,1])
  gc=geocode(end)
  longlat[j,]=as.matrix(gc)
  dataf[j,4]=longlat[j,2]
  dataf[j,5]=longlat[j,1]
}


writexl::write_xlsx(dataf, path = "../web_scraping/long_lat.xlsx")
