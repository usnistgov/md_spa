      SUBROUTINE STRUCTURE_FACTOR(SWITCH,L,NP,RXC,RYC,RZC)
      IMPLICIT NONE
      INCLUDE 'm3_global_rigid_2.f'
      DOUBLE PRECISION RXC(NPMAX), RYC(NPMAX), RZC(NPMAX)
      INTEGER  JQSQ, TMPX, TMPY, TMPZ, TMPSQ
      INTEGER  I, J, IQ
      DOUBLE PRECISION QR, SQCSUM, SQSSUM
      DOUBLE PRECISION L
      INTEGER  SWITCH, scounter

      INTEGER    NQALPHA, NQSQ, NQMAX
      PARAMETER (NQALPHA=30) 
      PARAMETER (NQSQ=NQALPHA*NQALPHA)
      PARAMETER (NQMAX=(2*NQALPHA+1)*(2*NQALPHA+1)*(2*NQALPHA+1))

      INTEGER   NKX, NKY, NKZ, NKMAX
      PARAMETER (NKX=6)
      PARAMETER (NKY=6)
      PARAMETER (NKZ=6)
      PARAMETER (NKMAX=(2*NKX+1)*(2*NKY+1)*(2*NKZ+1))

      INTEGER         IQX(NQMAX),  IQY(NQMAX), IQZ(NQMAX)
      INTEGER         IQSQ(NQMAX), NQ
      COMMON /SQINT/ IQX, IQY,        IQZ,
     :                IQSQ,        NQ

      REAL         SQ(NQSQ),    DEG(NQSQ),   SQNRM
      COMMON /SQREAL/ SQ,          DEG,         SQNRM

!     *****
!     IT CALCULATES THE STATIC STRUCTURE FACTOR FOR ALL THE CORES
!     *****

c     INITIALIZE S(q) : create histogram
      IF(SWITCH.EQ.0) THEN
         DO JQSQ = 1 , NQSQ
            DEG(JQSQ) = 0.
            SQ(JQSQ)  = 0.
         ENDDO
         SQNRM = 0.
         NQ = 0
         DO TMPX = 0 , NQALPHA
            DO TMPY = -NQALPHA, NQALPHA
               DO TMPZ = -NQALPHA, NQALPHA
                  TMPSQ = TMPX**2 + TMPY**2 + TMPZ**2
                  IF( TMPSQ .GT. 0 .AND. TMPSQ .LE. NQSQ ) THEN
                     NQ            = NQ + 1
                     IQX(NQ)       = TMPX
                     IQY(NQ)       = TMPY
                     IQZ(NQ)       = TMPZ
                     IQSQ(NQ)      = TMPSQ
                     IF( IQX(NQ) .EQ. 0 ) THEN
                        DEG(IQSQ(NQ)) = DEG(IQSQ(NQ)) + 1.0
                     ELSE
                        DEG(IQSQ(NQ)) = DEG(IQSQ(NQ)) + 2.0
                     ENDIF
                  ENDIF
               ENDDO
            ENDDO
         ENDDO

         IF( NQ .GT. NQMAX ) STOP 'VECTOR ARRAY ERROR'

      ELSE

c     UPDATE S(q) : update histogram
         IF(SWITCH.EQ.1) THEN
c            L = LX
c     LOOP OVER VECTORS

            DO IQ = 1, NQ
               SQCSUM = 0.0
               SQSSUM = 0.0

              write(*,*) INT(100*IQ/NQ), '%', IQ, NQ

c     TRIG SUMS
               DO I = 1, NP
                  QR = 2.0*PI/L*( REAL(IQX(IQ)) *RXC(I)
     :                                 +REAL(IQY(IQ)) *RYC(I)
     :                                 +REAL(IQZ(IQ)) *RZC(I))

                  SQCSUM = SQCSUM + COS(QR)
                  SQSSUM = SQSSUM + SIN(QR)
               ENDDO
c     SAVE SOME TIME
               IF( IQX(IQ) .EQ. 0 ) THEN
                  SQ(IQSQ(IQ)) = SQ(IQSQ(IQ))
     :                 + 1.0*(SQCSUM**2+SQSSUM**2)/REAL(np)
               ELSE
                  SQ(IQSQ(IQ)) = SQ(IQSQ(IQ))
     :                 + 2.0*(SQCSUM**2+SQSSUM**2)/REAL(np)
               ENDIF
            ENDDO
            SQNRM = SQNRM + 1.0

         ELSE

c     S(q) OUTPUT
c            L = LX
            OPEN(UNIT=44,FILE='output_SQ_cores.dat',STATUS='UNKNOWN')
            WRITE(44,4400) ISTEP
            WRITE(44,4410)
            DO JQSQ = 1 , NQSQ
               IF( DEG(JQSQ) .GT. 0.0 ) THEN
                  WRITE(44,4420) TWOPI*SQRT(REAL(JQSQ))/L,
     :                 SQ(JQSQ) / DEG(JQSQ) / SQNRM
               ENDIF
            ENDDO
            CLOSE(UNIT=44)

         ENDIF
      ENDIF

 4400 FORMAT('#',1X,'S(q) up to block',1X,I8)
 4410 FORMAT('#',1X,'     q*sigma',1X,'        S(q)')
 4420 FORMAT(    2X,         E12.6,1X,         E12.6)

      RETURN
      END
