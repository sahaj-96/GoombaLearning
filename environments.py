from multiprocessing import Pipe,Process

class Envcontrol:
    def controller(name,connection):
        while True:
            (command,args,kwargs)=connection.recv()
            if command=="reset":
                connection.send(env.reset())
            elif command=="step":
                connection.send(env.step(*args,**kwargs))
            elif command=="exit":
                break
            else:
                raise Exception(f"Unknown command {command}")
    def __init__(self,n):
        parent_side,child_side=Pipe()
        self.process=Process(target=Envcontrol.controller,args=(n,child_side))
        self.process.start()
        self.connection=parent_side
    def reset(self):
        self.connection.send(("reset",(),{}))
        return self.connection.recv()
    def step(self, action):
        self.connection.send(("step",(action,),{}))
        return self.connection.recv()
    def exit(self):
        self.connection.send(("exit",(),{}))
        self.process.join()

class TimeOrderedList:
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
