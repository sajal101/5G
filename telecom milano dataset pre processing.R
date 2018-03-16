inputDF <- read.csv(file="sms-call-internet-mi-2013-11-01.txt",sep="\t",header=F)


nrow(inputDF)

str(inputDF)

colnames(inputDF) <- c("square_id","timer_interval","country_code","sms_in_activity","sms_out_activity","call_in_activity","call_out_activity","internet_traffic_activity")
str(inputDF)
head(inputDF)


fn.deriveAdditionalFields <- function(inputDF){
  factorColumns <- c("square_id","country_code")
  inputDF[factorColumns] <- lapply(inputDF[factorColumns],as.factor)
  inputDF$activity_start_time <- fn.findStartTimeInterval(inputDF$timer_interval)
  inputDF$activity_date <- as.Date(as.POSIXct(inputDF$activity_start_time,origin="1970-01-01 "))
  inputDF$activity_time <- format(inputDF$activity_start_time,"%H")
  inputDF$total_activity <- rowSums(inputDF[, c(4,5,6,7,8)],na.rm=T)
  return(inputDF)
}
fn.findStartTimeInterval <- function(inputTime){
  val <- inputTime/1000
  outputTime <- as.POSIXct(val, origin="1970-01-01",tz="UTC")
  return(outputTime)
}


inputDF <- fn.deriveAdditionalFields(inputDF)


head(inputDF)


