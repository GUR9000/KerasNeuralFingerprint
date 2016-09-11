import os
# do not set RDBaseDir, so to trigger exceptions and track use:
# RDBaseDir=os.environ['RDBASE']
RDCodeDir=os.path.dirname(__file__)
_share = os.path.join(r'C:/Users/Gregor/Anaconda/Library', r'share/RDKit')
RDDataDir=os.path.join(_share,'Data')
RDDocsDir=os.path.join(_share,'Docs')
RDProjDir=os.path.join(_share,'Projects')
