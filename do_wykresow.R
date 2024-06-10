library(ggplot2)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
dane1 = data.frame(read.csv("dane_na_wykresy.csv",header = T))
dane1$power <-as.numeric(dane1$power)
dane1$length <-as.numeric(dane1$length)

ggplot(dane1,aes(x=length,y=power,col = model))+geom_line()



