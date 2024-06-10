library(ggplot2)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
dane1 = data.frame(read.csv("dane_na_wykresy.csv",header = T))
dane1$power <-as.numeric(dane1$power)
dane1$length <-as.numeric(dane1$length)

ggplot(dane1,aes(x=length,y=power,col = model))+geom_line()+   
  labs(x = "Length of the examined data", y = "Power", title = "Power for length of the examined data",
       subtitle = "for different models")+
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "bottom")


dane2 = data.frame(read.csv("dane_na_wykresy_2.csv",header = T))

dane2$n <-as.numeric(dane2$n)
dane2$value <-as.numeric(dane2$value)

ggplot(dane2,aes(x=n,y=value,col = device))+geom_line()+facet_wrap(~kryt,ncol = 1)+   
  labs(x = "Number of hidden states", y = "Criterion", title = "Value of criterions depending on the nr. of hidden states",
       subtitle = "for different devices")+
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "bottom")


prady <-(read.csv("house3_5devices_train.csv",header = T))
n <- length(prady[,1])
dan<-rbind(
  cbind(1:n,prady[,2],"lighting2"),
  cbind(1:n,prady[,3],"lighting5"),
  cbind(1:n,prady[,4],"lighting4"),
  cbind(1:n,prady[,5],"refrigerator"),
  cbind(1:n,prady[,6],"microwave")
)
prady<-data.frame(dan)
colnames(prady)<-c("time","amount","device")
prady$time <-as.numeric(prady$time)
prady$amount <-as.numeric(prady$amount)

ggplot(subset(prady,prady$device!="refrigerator" & prady$device!="microwave"),
       aes(x=time,y=amount,col = device))+geom_line()

ggplot(subset(prady,prady$device=="refrigerator"),
       aes(x=time,y=amount))+geom_line(col = "purple")

ggplot(subset(prady,prady$device=="microwave"),
       aes(x=time,y=log(amount)))+geom_line(col = "red")











