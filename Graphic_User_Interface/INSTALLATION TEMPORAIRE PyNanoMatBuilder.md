INSTALLATION TEMPORAIRE PyNanoMatBuilderViewer
-------------------------------------------------------
- Télécharger la derniere version de PyNanoMatBuilder sur github
- copier dans le dossier site package de l'environnement (ici ai_py311) C:\Users\cayez\Anaconda3\envs\ai_py311\Lib\site-packages 
1) le dossier figs
2) créer un dossier pyNanoMatBuilder et coller dans ce dossier les fichiers contenus dans src sur github
Dans le code les appels aux fonctions de génération de particules se feront avec la forme 
self.MyNP = pNP.regIco(self.atom,self.size0, nShell=int(self.size1),aseView=False,thresholdCoreSurface = 0.,skipSymmetryAnalyzis = True,noOutput = True)