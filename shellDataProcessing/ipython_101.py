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