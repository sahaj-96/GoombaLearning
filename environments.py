class TimeOrderedList(object):
    def __init__(self,first_time=0):
        self.first_time=first_time
        self.list=[]
    def remove_excess(self,time_interval):
        for_removing=time_interval-self.first_time+1
        if for_removing>0:
            self.list=self.list[for_removing:]
            self.first_time=time_interval+1
    def append(self,to_add):
        self.list.append(to_add)
    def get(self, at_time_t):
        return self.list[at_time_t-self.first_time]
    def future_len(self):
        return len(self.list)

    def in_range(self,from_time,length):
        return self.list[(from_time-self.first_time):(from_time-self.first_time+length)]
