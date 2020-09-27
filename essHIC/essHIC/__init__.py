print('''
	 ##############################################################################################################################
         ##########                                                                                                                   #
	 ######:::##                                                                                                                  #
	 ####::::::##                                                 HHHHHHHHH     HHHHHHHHHIIIIIIIIII      CCCCCCCCCCCCC            #
	 ##:::::::::##                                                H:::::::H     H:::::::HI::::::::I   CCC::::::::::::C            #
	 ##############	                                              H:::::::H     H:::::::HI::::::::I CC:::::::::::::::C            #
	 #	                                                      HH::::::H     H::::::HHII::::::IIC:::::CCCCCCCC::::C            #
	 #	    eeeeeeeeeeee        ssssssssss       ssssssssss     H:::::H     H:::::H    I::::I C:::::C       CCCCCC            #
	 #	  ee::::::::::::ee    ss::::::::::s    ss::::::::::s    H:::::H     H:::::H    I::::IC:::::C                          #
	 #	 e::::::eeeee:::::eess:::::::::::::s ss:::::::::::::s   H::::::HHHHH::::::H    I::::IC:::::C                          #
	 #	e::::::e     e:::::es::::::ssss:::::ss::::::ssss:::::s  H:::::::::::::::::H    I::::IC:::::C                          #
	 #	e:::::::eeeee::::::e s:::::s  ssssss  s:::::s  ssssss   H:::::::::::::::::H    I::::IC:::::C                          #
	 #	e:::::::::::::::::e    s::::::s         s::::::s        H::::::HHHHH::::::H    I::::IC:::::C                          #
	 #	e::::::eeeeeeeeeee        s::::::s         s::::::s     H:::::H     H:::::H    I::::IC:::::C                          #
	 #	e:::::::e           ssssss   s:::::s ssssss   s:::::s   H:::::H     H:::::H    I::::I C:::::C       CCCCCC            #
	 #	e::::::::e          s:::::ssss::::::ss:::::ssss::::::sHH::::::H     H::::::HHII::::::IIC:::::CCCCCCCC::::C            #
	 #	 e::::::::eeeeeeee  s::::::::::::::s s::::::::::::::s H:::::::H     H:::::::HI::::::::I CC:::::::::::::::C            #
	 #	  ee:::::::::::::e   s:::::::::::ss   s:::::::::::ss  H:::::::H     H:::::::HI::::::::I   CCC::::::::::::C            #
	 #	    eeeeeeeeeeeeee    sssssssssss      sssssssssss    HHHHHHHHH     HHHHHHHHHIIIIIIIIII      CCCCCCCCCCCCC            #
         #                                                                                                                            #
         #                                                                                                                            #
         ##############################################################################################################################''')
         

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from essHIC.make_hic import make_hic
from essHIC.hic import hic
from essHIC.hic import pseudo
from essHIC.ess import ess
from essHIC.dist import dist
from essHIC.utils import *
