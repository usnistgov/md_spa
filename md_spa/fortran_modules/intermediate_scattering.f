      SUBROUTINE INTERMEDIATE_SCATTERING(RXC,RYC,RZC,NP,L,switch)
      IMPLICIT NONE
      INCLUDE 'm3_global_rigid_2.f'
C       * INTEGER PARAMETERS *
      INTEGER    NNMAX,NTCFMAX, NKMAX
      PARAMETER (NNMAX=500)
      PARAMETER (NTCFMAX=2100)
      PARAMETER (NKMAX=120)

C * TCF DYNAMICAL STUFF *
      INTEGER ITCF, JTCF, KTCF
      REAL  MSDTOTAL
      REAL  RXZERO(NNMAX), RYZERO(NNMAX), RZZERO(NNMAX)
      REAL  RXTCF(NNMAX,0:NTCFMAX), RYTCF(NNMAX,0:NTCFMAX),
     :            RZTCF(NNMAX,0:NTCFMAX)
      REAL        DRXDTTCF(NNMAX,0:NTCFMAX), DRYDTTCF(NNMAX,0:NTCFMAX),
     :            DRZDTTCF(NNMAX,0:NTCFMAX)
      REAL  COSKRI(NNMAX,NKMAX)
      REAL  COSKRITCF(NNMAX,NKMAX,0:NTCFMAX)
      REAL  SINKRI(NNMAX,NKMAX)
      REAL  SINKRITCF(NNMAX,NKMAX,0:NTCFMAX)
      REAL  ATCF(NKMAX,0:NTCFMAX)
      REAL  BTCF(NKMAX,0:NTCFMAX)
      REAL  JXATCF(NKMAX,0:NTCFMAX)
      REAL  JYATCF(NKMAX,0:NTCFMAX)
      REAL  JZATCF(NKMAX,0:NTCFMAX)
      REAL        JXBTCF(NKMAX,0:NTCFMAX)
      REAL        JYBTCF(NKMAX,0:NTCFMAX)
      REAL        JZBTCF(NKMAX,0:NTCFMAX)
      REAL        ZETAXX,  ZETAXY
      REAL ZETAYX,  ZETAYY
      REAL ZETAXYTCF(0:NTCFMAX)
      COMMON /DEIGHT/ ITCF, JTCF, KTCF,
     : MSDTOTAL,
     :                  RXTCF,    RYTCF,    RZTCF,
     :                  DRXDTTCF, DRYDTTCF, DRZDTTCF,
     : COSKRI,
     : COSKRITCF,
     : SINKRI,
     : SINKRITCF,
     :                  ATCF,
     :                  BTCF,
     : JXATCF,
     : JYATCF, JZATCF,
     : JXBTCF,
     : JYBTCF, JZBTCF,
     :                  ZETAXX, ZETAXY,
     : ZETAYX, ZETAYY,
     : ZETAXYTCF

C * RECIPROCAL SPACE *
      INTEGER IK,         JK,         NK
      INTEGER NKX(NKMAX), NKY(NKMAX), NKZ(NKMAX), NKSQ(NKMAX)
        REAL KX(NKMAX),  KY(NKMAX), KZ(NKMAX),
     :                  KSQ(NKMAX), KABS(NKMAX)
      COMMON /RPR/ IK,         JK,         NK,
     :                  NKX,        NKY,        NKZ, NKSQ,
     :                  KX,         KY,         KZ,
     :                  KSQ,        KABS
      LOGICAL     TCFFULL
      COMMON /LOGIC/  TCFFULL

C * BIG DYNAMICAL ACCUMULATORS
      DOUBLEPRECISION MSD(0:NTCFMAX)
      DOUBLEPRECISION VACF(0:NTCFMAX)
      DOUBLEPRECISION REFS(0:NTCFMAX)
      DOUBLEPRECISION REFA(0:NTCFMAX)
      DOUBLEPRECISION RECL(0:NTCFMAX)
      DOUBLEPRECISION RECT(0:NTCFMAX)
      DOUBLEPRECISION ETAS(0:NTCFMAX)
      DOUBLEPRECISION TCFNORM(0:NTCFMAX)
      COMMON /DYNAM/ MSD,
     : VACF,
     : REFS,
     :                  REFA,
     :                  RECL,
     :                  RECT,
     : ETAS,
     : TCFNORM


      REAL KRI
      REAL JLAITCF, JLBITCF, JTAITCF, JTBITCF
      REAL JLAJTCF, JLBJTCF, JTAJTCF, JTBJTCF

      INTEGER NTCF

      REAL     L, LO2
      integer   irange
      real      crossx(npmax), crossy(npmax), crossz(npmax)
      DOUBLE PRECISION rxij, ryij, rzij
      integer   q_peak

      DOUBLE PRECISION rxc(npmax),
     :          ryc(npmax),
     :          rzc(npmax)


      integer i, n, switch

      integer nqalphaa, tmpx,tmpy,tmpz,TMPSQ

!     *****
!     IT CALCULATES THE INTERMEDIATE SCATTERING FUNCTION
!
!     F (q, t) =  1 / N * < rho_k (t) rho_-k(0) >
!
!     *****

      NTCF = 2000
      n = np
c      l = lx
      lo2 = l /2.
      qdes = 600
      if (switch.eq.0) then

      nk = 0
      NQALPHAa = 30

      DO TMPX = 0, NQALPHAa
         DO TMPY = -NQALPHAa, NQALPHAa
            DO TMPZ = -NQALPHAa, NQALPHAa
               TMPSQ = TMPX**2 + TMPY**2 + TMPZ**2
               IF( TMPSQ .eq. qdes ) THEN
                  NK            = NK + 1
                  NKX(Nk)       = TMPX
                  NKY(Nk)       = TMPY
                  NKZ(Nk)       = TMPZ
               ENDIF
            ENDDO
         ENDDO
      ENDDO

      if (nk.gt.nkmax) stop 'nk error'

C * RECIPROCAL SPACE VECTORS *
      DO IK = 1 , NK
         KX(IK)   = TWOPI * REAL(NKX(IK)) / L
         KY(IK)   = TWOPI * REAL(NKY(IK)) / L
         KZ(IK)   = TWOPI * REAL(NKZ(IK)) / L
         NKSQ(IK) = NKX(IK)*NKX(IK) + NKY(IK)*NKY(IK)+ NKZ(IK)*NKZ(IK)
         KSQ(IK)  = KX(IK) *KX(IK)  + KY(IK) *KY(IK) + KZ(IK) *KZ(IK)
         KABS(IK) = SQRT(KSQ(IK))
      ENDDO
      DO IK = 1 , NK
         DO JK = 1 , NK
            IF( NKSQ(IK) .NE. NKSQ(JK) ) STOP 'KSQ ERROR'
         ENDDO
      ENDDO
     
C * ZERO GLOBAL TCF COUNTERS (d.p.) *
      DO KTCF = 0 , NTCF
         REFS(KTCF)     = 0.0D0
         REFA(KTCF)     = 0.0D0
         TCFNORM(KTCF)  = 0.0D0
      ENDDO

C       * INTERMEDIATE SCATTERING FUNCTION *
      DO I = 1 , N
         DO IK = 1, NK
            KRI          = KX(IK)*RXC(I) + KY(IK)*RYC(I)+ KZ(IK)*RZC(I)
            COSKRI(I,IK) = COS(KRI)
            SINKRI(I,IK) = SIN(KRI)
         ENDDO
      ENDDO

C * EMPTY BUFFERS TO START *
      TCFFULL = .FALSE.
      ITCF = 0

C * PUT IN 0th VALUES *
      DO IK = 1 , NK
         ATCF(IK,ITCF) = 0.0
         BTCF(IK,ITCF) = 0.0
      ENDDO
      DO I = 1, N
         RXTCF(I,ITCF)    = RXC(I)
         RYTCF(I,ITCF)    = RYC(I)
         RZTCF(I,ITCF)    = RZC(I)
         DO IK = 1 , NK
               COSKRITCF(I,IK,ITCF) = COSKRI(I,IK)
               SINKRITCF(I,IK,ITCF) = SINKRI(I,IK)
               ATCF(IK,ITCF) = ATCF(IK,ITCF) + COSKRI(I,IK)
               BTCF(IK,ITCF) = BTCF(IK,ITCF) + SINKRI(I,IK)
         ENDDO
      ENDDO

      else  !======================================= TCFinitial
      if (switch.eq.1)then

C       * INTERMEDIATE SCATTERING FUNCTION *
      DO I = 1 , N
         DO IK = 1, NK
           KRI = KX(IK)*RXC(I) + KY(IK)*RYC(I) + KZ(IK)*RZC(I)
           COSKRI(I,IK) = COS(KRI)
           SINKRI(I,IK) = SIN(KRI)
         ENDDO
      ENDDO

C       * TIME CORRELATION FUNCTIONS *
C * IF NTCF IS ALREADY FULL, MAKE SPACE *
      IF( TCFFULL ) THEN
         DO JTCF = 0 , ITCF-1
             DO I = 1 , N
                RXTCF(I,JTCF)    = RXTCF(I,JTCF+1)
                RYTCF(I,JTCF)    = RYTCF(I,JTCF+1)
                RZTCF(I,JTCF)    = RZTCF(I,JTCF+1)
                DRXDTTCF(I,JTCF) = DRXDTTCF(I,JTCF+1)
                DRYDTTCF(I,JTCF) = DRYDTTCF(I,JTCF+1)
                DRZDTTCF(I,JTCF) = DRZDTTCF(I,JTCF+1)
                DO IK = 1 , NK
                   COSKRITCF(I,IK,JTCF) = COSKRITCF(I,IK,JTCF+1)
                   SINKRITCF(I,IK,JTCF) = SINKRITCF(I,IK,JTCF+1)
                ENDDO
             ENDDO
             DO IK = 1 , NK
                ATCF(IK,JTCF)   = ATCF(IK,JTCF+1)
                BTCF(IK,JTCF)   = BTCF(IK,JTCF+1)
             ENDDO
         ENDDO
      ENDIF

C * INCREMENT COUNTER AND CHECK IF FULL *
      IF( .NOT. TCFFULL ) THEN
         ITCF = ITCF + 1
      IF( ITCF .EQ. NTCF ) TCFFULL = .TRUE.
      ELSE
         ITCF = NTCF
      ENDIF

C * NOW PUT IN NEW VALUES *
      DO IK = 1 , NK
         ATCF(IK,ITCF)   = 0.
         BTCF(IK,ITCF)   = 0.
      ENDDO

      DO I = 1 , N
         RXTCF(I,ITCF)    = RXC(I)
         RYTCF(I,ITCF)    = RYC(I)
         RZTCF(I,ITCF)    = RZC(I)
         DO IK = 1 , NK
            COSKRITCF(I,IK,ITCF) = COSKRI(I,IK)
            SINKRITCF(I,IK,ITCF) = SINKRI(I,IK)
            ATCF(IK,ITCF) = ATCF(IK,ITCF) + COSKRI(I,IK)
            BTCF(IK,ITCF) = BTCF(IK,ITCF) + SINKRI(I,IK)
         ENDDO
      ENDDO
! ZETAXYTCF(ITCF) = ZETAXY

C * NOW UPDATE TIME CORRELATION FUNCTIONS *
C * N.B. t(JTCF) < t(ITCF)                *

      DO JTCF = 0 , ITCF
         KTCF = ITCF - JTCF
         IF( KTCF .GE. 0 .AND. KTCF .LE. NTCF ) THEN
             TCFNORM(KTCF) = TCFNORM(KTCF) + 1.0D0
             DO I = 1 , N
C          * DIFFUSION *
                MSD(KTCF)  = MSD(KTCF)
     :                + DBLE(RXTCF(I,ITCF)-RXTCF(I,JTCF))**2
     :                + DBLE(RYTCF(I,ITCF)-RYTCF(I,JTCF))**2
     :                + DBLE(RZTCF(I,ITCF)-RZTCF(I,JTCF))**2
C          * VELOCITY AUTOCORRELATION FUNCTION *
                VACF(KTCF) = VACF(KTCF)
     :                + DBLE(DRXDTTCF(I,ITCF)*DRXDTTCF(I,JTCF))
     :                + DBLE(DRYDTTCF(I,ITCF)*DRYDTTCF(I,JTCF))
     :                + DBLE(DRZDTTCF(I,ITCF)*DRZDTTCF(I,JTCF))

                DO IK = 1 , NK
C          * INCOHERENT SCATTERING FUNCTION *
                   REFS(KTCF) = REFS(KTCF)
     :      + DBLE(COSKRITCF(I,IK,ITCF)*COSKRITCF(I,IK,JTCF))
     :      + DBLE(SINKRITCF(I,IK,ITCF)*SINKRITCF(I,IK,JTCF))
                ENDDO
             ENDDO

             DO IK = 1 , NK
C   * COHERENT SCATTERING FUNCTION *
                REFA(KTCF) = REFA(KTCF)
     :                + DBLE(ATCF(IK,ITCF)*ATCF(IK,JTCF))
     :                + DBLE(BTCF(IK,ITCF)*BTCF(IK,JTCF))
             ENDDO

         ENDIF
      ENDDO

      else !===================================================== Calc

      OPEN(UNIT=40,FILE='data.TCF',STATUS='UNKNOWN')
      WRITE(40,4000)
      DO KTCF = 0 , NTCF
         WRITE(40,4010) KTCF,
     :                  REAL(KTCF*NCALC)*DT,
     :                  MSD(KTCF) / TCFNORM(KTCF) / DBLE(N),
     :                  VACF(KTCF) / TCFNORM(KTCF) / DBLE(N),
     :                  KABS(1),
     :                  REFS(KTCF) / TCFNORM(KTCF) / DBLE(N*NK),
     :                  REFA(KTCF) / TCFNORM(KTCF) / DBLE(N*NK)
      ENDDO
      CLOSE(UNIT=40)

      endif
      endif

 4000 FORMAT('#',1X,    '    Step',
     :             1X,'        Time',
     :             1X,'    <R^2(t)>',
     :             1X,  '    Cvv(t)',
     :             1X,  '         k',
     :             1X,  '   Fs(k,t)',
     :             1X,  '   Fa(k,t)')
 4010 FORMAT(    2X,            I8,
     :             1X,         E12.6,
     :             1X,         E12.6,
     :             1X,         E10.4,
     :             1X,         E10.4,
     :             1X,         E10.4,
     :             1X,         E10.4)

      RETURN
      END
