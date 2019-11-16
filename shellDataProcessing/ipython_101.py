#!/usr/bin/env python


# ipython command line

!df -h
!python -c "from random import choices;days = ['Mo', 'Tu', 'We', 'Th', 'Fr'];print(choices(days))"

ls = !ls
type(ls)

var = !ls -h *.csv
print(len(ls))


ls_list = !ls -l |grep .py| awk '{ SUM+=$5} END {print SUM }'
type(ls_list)
ls_list

####################################################################
# using the bash to give the output  to stdoutof series of commands
%%bash --out output
ls -l |grep .py| awk '{ SUM+=$5} END {print SUM}'


# captur the error to stderr
%%bash --out output --err outerr
ls -l |grep .py| awk '{ SUM+=$5} END {print SUM}'
echo 'No error ' >&2

# get the output
type(output)
output
########################################################################################


# # below is python intepreter throuch the shell
# python -c "from random import choices;days = ['Mo', 'Tu', 'We', 'Th', 'Fr'];print(choices(days))"

# python -c "import datetime;print(datetime.datetime.utcnow())"

# below is unix command
# ls -l | awk '{ SUM+=$5} END {print SUM}'
# ls -l | grep .py| awk '{ SUM+=$5} END {print SUM}'


# SLIST DATATYPE #################################################
# SLIST METHODS: fields, grep, sort
mylist = !ls -h
mylist ## the same as mylist.l, the default, list
mylist.s ## space separated string
mylist.n ## newline, \n separated string
mylist.p  ## list of file path

# fields splits the output into whitespace delimited columns 
# and returns the values of columns, specified by their indices, as space-separated strings.
# split the string into columns delimited by space
# you can do now do further slicing using indices, (1,4) for columns and [1:3] for rows
myls = !ls -l 
all_folder_sub = !ls -R
myls.fields(2,4)[1:3]


# grep search for patterns
myls.grep('kill')

# search for file ending in .py
all_folder_sub.grep('(\.py)$')



# sort the output by column index (first argument) and optional numerical (nums=True)
myls.sort(5, nums=True)

# convert to other datatypes
myset = set(myls)
mylist2 = [myls]
mydict1 = dict(vals=myls)
mydict2 = {"vals2":myls}
