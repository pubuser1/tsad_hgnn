
#------------- Print total training time ---------------------------------------------
def print_train_time(train_time):
 precision = 2
 time_measure = ''
 time = 0.0
 
 if train_time>3600:
  time = train_time / 3600
  time_measure = 'hrs'
 elif train_time>60:
  time = train_time / 60
  time_measure = 'mins'
 else:
  time = train_time
  time_measure = 'secs'
  
 formatted_time = f"{time:.{precision}f}"
 print('Training time (',time_measure,') : ',formatted_time) 
 
#------------- Print total inference time ---------------------------------------------
def get_time(test_time):
 precision = 2
 time_measure = ''
 time = 0.0
 
 if test_time>3600:
  time = test_time / 3600
  time_measure = 'hrs'
 elif test_time>60:
  time = test_time / 60
  time_measure = 'mins'
 else:
  time = test_time
  time_measure = 'secs'
  
 formatted_time = f"{time:.{precision}f}"
 return time_measure, formatted_time
#print('Inference time (',time_measure,') : ',formatted_time)
