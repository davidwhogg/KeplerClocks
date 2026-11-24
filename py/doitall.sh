# don't execute this file. Just `cat` it and then copy-and-paste.
python clocks.py db
nohup python clocks.py worker > worker0.log 2>&1 &
nohup python clocks.py worker > worker1.log 2>&1 &
nohup python clocks.py worker > worker2.log 2>&1 &
nohup python clocks.py worker > worker3.log 2>&1 &
nohup python clocks.py worker > worker4.log 2>&1 &
nohup python clocks.py worker > worker5.log 2>&1 &
nohup python clocks.py worker > worker6.log 2>&1 &
caffeinate nohup python clocks.py worker > worker7.log 2>&1 &
